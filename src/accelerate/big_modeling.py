# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import re
from contextlib import contextmanager
from functools import wraps
from typing import Optional, Union

import torch
import torch.nn as nn

from .hooks import (
    AlignDevicesHook,
    CpuOffload,
    LayerwiseCastingHook,
    UserCpuOffloadHook,
    add_hook_to_module,
    attach_align_device_hook,
    attach_align_device_hook_on_blocks,
)
from .utils import (
    OffloadedWeightsLoader,
    check_cuda_p2p_ib_support,
    check_device_map,
    extract_submodules_state_dict,
    find_tied_parameters,
    get_balanced_memory,
    infer_auto_device_map,
    is_bnb_available,
    is_mlu_available,
    is_musa_available,
    is_npu_available,
    is_sdaa_available,
    is_xpu_available,
    load_checkpoint_in_model,
    offload_state_dict,
    parse_flag_from_env,
    retie_parameters,
)
from .utils.constants import SUPPORTED_PYTORCH_LAYERS_FOR_UPCASTING
from .utils.other import recursive_getattr


logger = logging.getLogger(__name__)


@contextmanager
def init_empty_weights(include_buffers: bool = None):
    """
    A context manager under which models are initialized with all parameters on the meta device, therefore creating an
    empty model. Useful when just initializing the model would blow the available RAM.

    Args:
        include_buffers (`bool`, *optional*):
            Whether or not to also put all buffers on the meta device while initializing.

    Example:

    ```python
    import torch.nn as nn
    from accelerate import init_empty_weights

    # Initialize a model with 100 billions parameters in no time and without using any RAM.
    with init_empty_weights():
        tst = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
    ```

    <Tip warning={true}>

    Any model created under this context manager has no weights. As such you can't do something like
    `model.to(some_device)` with it. To load weights inside your empty model, see [`load_checkpoint_and_dispatch`].
    Make sure to overwrite the default device_map param for [`load_checkpoint_and_dispatch`], otherwise dispatch is not
    called.

    </Tip>
    """
    if include_buffers is None:
        include_buffers = parse_flag_from_env("ACCELERATE_INIT_INCLUDE_BUFFERS", False)
    with init_on_device(torch.device("meta"), include_buffers=include_buffers) as f:
        yield f


@contextmanager
def init_on_device(device: torch.device, include_buffers: bool = None):
    """
    A context manager under which models are initialized with all parameters on the specified device.

    Args:
        device (`torch.device`):
            Device to initialize all parameters on.
        include_buffers (`bool`, *optional*):
            Whether or not to also put all buffers on the meta device while initializing.

    Example:

    ```python
    import torch.nn as nn
    from accelerate import init_on_device

    with init_on_device(device=torch.device("cuda")):
        tst = nn.Linear(100, 100)  # on `cuda` device
    ```
    """
    if include_buffers is None:
        include_buffers = parse_flag_from_env("ACCELERATE_INIT_INCLUDE_BUFFERS", False)

    if include_buffers:
        with device:
            yield
        return

    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    # Patch tensor creation
    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    try:
        nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


def cpu_offload(
    model: nn.Module,
    execution_device: Optional[torch.device] = None,
    offload_buffers: bool = False,
    state_dict: Optional[dict[str, torch.Tensor]] = None,
    preload_module_classes: Optional[list[str]] = None,
):
    """
    Activates full CPU offload for a model. As a result, all parameters of the model will be offloaded and only one
    copy of the state dict of the model will be kept. During the forward pass, parameters will be extracted from that
    state dict and put on the execution device passed as they are needed, then offloaded again.

    Args:
        model (`torch.nn.Module`):
            The model to offload.
        execution_device (`torch.device`, *optional*):
            The device on which the forward pass of the model will be executed (should be a GPU). Will default to the
            model first parameter device.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to offload the buffers with the model parameters.
        state_dict (`Dict[str, torch.Tensor]`, *optional*):
            The state dict of the model that will be kept on CPU.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.
    """
    if execution_device is None:
        execution_device = next(iter(model.parameters())).device
    if state_dict is None:
        state_dict = {n: p.to("cpu") for n, p in model.state_dict().items()}

    add_hook_to_module(model, AlignDevicesHook(io_same_device=True), append=True)
    attach_align_device_hook(
        model,
        execution_device=execution_device,
        offload=True,
        offload_buffers=offload_buffers,
        weights_map=state_dict,
        preload_module_classes=preload_module_classes,
    )

    return model


def cpu_offload_with_hook(
    model: torch.nn.Module,
    execution_device: Optional[Union[int, str, torch.device]] = None,
    prev_module_hook: Optional[UserCpuOffloadHook] = None,
):
    """
    Offloads a model on the CPU and puts it back to an execution device when executed. The difference with
    [`cpu_offload`] is that the model stays on the execution device after the forward and is only offloaded again when
    the `offload` method of the returned `hook` is called. Useful for pipelines running a model in a loop.

    Args:
        model (`torch.nn.Module`):
            The model to offload.
        execution_device(`str`, `int` or `torch.device`, *optional*):
            The device on which the model should be executed. Will default to the MPS device if it's available, then
            GPU 0 if there is a GPU, and finally to the CPU.
        prev_module_hook (`UserCpuOffloadHook`, *optional*):
            The hook sent back by this function for a previous model in the pipeline you are running. If passed, its
            offload method will be called just before the forward of the model to which this hook is attached.

    Example:

    ```py
    model_1, hook_1 = cpu_offload_with_hook(model_1, cuda_device)
    model_2, hook_2 = cpu_offload_with_hook(model_2, cuda_device, prev_module_hook=hook_1)
    model_3, hook_3 = cpu_offload_with_hook(model_3, cuda_device, prev_module_hook=hook_2)

    hid_1 = model_1(input)
    for i in range(50):
        # model1 is offloaded on the CPU at the first iteration, model 2 stays on the GPU for this whole loop.
        hid_2 = model_2(hid_1)
    # model2 is offloaded to the CPU just before this forward.
    hid_3 = model_3(hid_3)

    # For model3, you need to manually call the hook offload method.
    hook_3.offload()
    ```
    """
    hook = CpuOffload(execution_device=execution_device, prev_module_hook=prev_module_hook)
    add_hook_to_module(model, hook, append=True)
    user_hook = UserCpuOffloadHook(model, hook)
    return model, user_hook


def disk_offload(
    model: nn.Module,
    offload_dir: Union[str, os.PathLike],
    execution_device: Optional[torch.device] = None,
    offload_buffers: bool = False,
    preload_module_classes: Optional[list[str]] = None,
):
    """
    Activates full disk offload for a model. As a result, all parameters of the model will be offloaded as
    memory-mapped array in a given folder. During the forward pass, parameters will be accessed from that folder and
    put on the execution device passed as they are needed, then offloaded again.

    Args:
        model (`torch.nn.Module`): The model to offload.
        offload_dir (`str` or `os.PathLike`):
            The folder in which to offload the model weights (or where the model weights are already offloaded).
        execution_device (`torch.device`, *optional*):
            The device on which the forward pass of the model will be executed (should be a GPU). Will default to the
            model's first parameter device.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to offload the buffers with the model parameters.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.
    """
    if not os.path.isdir(offload_dir) or not os.path.isfile(os.path.join(offload_dir, "index.json")):
        offload_state_dict(offload_dir, model.state_dict())
    if execution_device is None:
        execution_device = next(iter(model.parameters())).device
    weights_map = OffloadedWeightsLoader(save_folder=offload_dir)

    add_hook_to_module(model, AlignDevicesHook(io_same_device=True), append=True)
    attach_align_device_hook(
        model,
        execution_device=execution_device,
        offload=True,
        offload_buffers=offload_buffers,
        weights_map=weights_map,
        preload_module_classes=preload_module_classes,
    )

    return model


def dispatch_model(
    model: nn.Module,
    device_map: dict[str, Union[str, int, torch.device]],
    main_device: Optional[torch.device] = None,
    state_dict: Optional[dict[str, torch.Tensor]] = None,
    offload_dir: Optional[Union[str, os.PathLike]] = None,
    offload_index: Optional[dict[str, str]] = None,
    offload_buffers: bool = False,
    skip_keys: Optional[Union[str, list[str]]] = None,
    preload_module_classes: Optional[list[str]] = None,
    force_hooks: bool = False,
):
    """
    Dispatches a model according to a given device map. Layers of the model might be spread across GPUs, offloaded on
    the CPU or even the disk.

    Args:
        model (`torch.nn.Module`):
            The model to dispatch.
        device_map (`Dict[str, Union[str, int, torch.device]]`):
            A dictionary mapping module names in the models `state_dict` to the device they should go to. Note that
            `"disk"` is accepted even if it's not a proper value for `torch.device`.
        main_device (`str`, `int` or `torch.device`, *optional*):
            The main execution device. Will default to the first device in the `device_map` different from `"cpu"` or
            `"disk"`.
        state_dict (`Dict[str, torch.Tensor]`, *optional*):
            The state dict of the part of the model that will be kept on CPU.
        offload_dir (`str` or `os.PathLike`):
            The folder in which to offload the model weights (or where the model weights are already offloaded).
        offload_index (`Dict`, *optional*):
            A dictionary from weight name to their information (`dtype`/ `shape` or safetensors filename). Will default
            to the index saved in `save_folder`.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to offload the buffers with the model parameters.
        skip_keys (`str` or `List[str]`, *optional*):
            A list of keys to ignore when moving inputs or outputs between devices.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.
        force_hooks (`bool`, *optional*, defaults to `False`):
            Whether or not to force device hooks to be attached to the model even if all layers are dispatched to a
            single device.
    """
    # Error early if the device map is incomplete.
    check_device_map(model, device_map)

    # We need to force hook for quantized model that can't be moved with to()
    if getattr(model, "quantization_method", "bitsandbytes") == "bitsandbytes":
        # since bnb 0.43.2, we can move 4-bit model
        if getattr(model, "is_loaded_in_8bit", False) or (
            getattr(model, "is_loaded_in_4bit", False) and not is_bnb_available(min_version="0.43.2")
        ):
            force_hooks = True

    # We attach hooks if the device_map has at least 2 different devices or if
    # force_hooks is set to `True`. Otherwise, the model in already loaded
    # in the unique device and the user can decide where to dispatch the model.
    # If the model is quantized, we always force-dispatch the model
    if (len(set(device_map.values())) > 1) or force_hooks:
        if main_device is None:
            if set(device_map.values()) == {"cpu"} or set(device_map.values()) == {"cpu", "disk"}:
                main_device = "cpu"
            else:
                main_device = [d for d in device_map.values() if d not in ["cpu", "disk"]][0]

        if main_device != "cpu":
            cpu_modules = [name for name, device in device_map.items() if device == "cpu"]
            if state_dict is None and len(cpu_modules) > 0:
                state_dict = extract_submodules_state_dict(model.state_dict(), cpu_modules)

        disk_modules = [name for name, device in device_map.items() if device == "disk"]
        if offload_dir is None and offload_index is None and len(disk_modules) > 0:
            raise ValueError(
                "We need an `offload_dir` to dispatch this model according to this `device_map`, the following submodules "
                f"need to be offloaded: {', '.join(disk_modules)}."
            )
        if (
            len(disk_modules) > 0
            and offload_index is None
            and (not os.path.isdir(offload_dir) or not os.path.isfile(os.path.join(offload_dir, "index.json")))
        ):
            disk_state_dict = extract_submodules_state_dict(model.state_dict(), disk_modules)
            offload_state_dict(offload_dir, disk_state_dict)

        execution_device = {
            name: main_device if device in ["cpu", "disk"] else device for name, device in device_map.items()
        }
        execution_device[""] = main_device
        offloaded_devices = ["disk"] if main_device == "cpu" or main_device == "mps" else ["cpu", "disk"]
        offload = {name: device in offloaded_devices for name, device in device_map.items()}
        save_folder = offload_dir if len(disk_modules) > 0 else None
        if state_dict is not None or save_folder is not None or offload_index is not None:
            device = main_device if offload_index is not None else None
            weights_map = OffloadedWeightsLoader(
                state_dict=state_dict, save_folder=save_folder, index=offload_index, device=device
            )
        else:
            weights_map = None

        # When dispatching the model's parameters to the devices specified in device_map, we want to avoid allocating memory several times for the
        # tied parameters. The dictionary tied_params_map keeps track of the already allocated data for a given tied parameter (represented by its
        # original pointer) on each devices.
        tied_params = find_tied_parameters(model)

        tied_params_map = {}
        for group in tied_params:
            for param_name in group:
                # data_ptr() is enough here, as `find_tied_parameters` finds tied params simply by comparing `param1 is param2`, so we don't need
                # to care about views of tensors through storage_offset.
                data_ptr = recursive_getattr(model, param_name).data_ptr()
                tied_params_map[data_ptr] = {}

                # Note: To handle the disk offloading case, we can not simply use weights_map[param_name].data_ptr() as the reference pointer,
                # as we have no guarantee that safetensors' `file.get_tensor()` will always give the same pointer.

        attach_align_device_hook_on_blocks(
            model,
            execution_device=execution_device,
            offload=offload,
            offload_buffers=offload_buffers,
            weights_map=weights_map,
            skip_keys=skip_keys,
            preload_module_classes=preload_module_classes,
            tied_params_map=tied_params_map,
        )

        # warn if there is any params on the meta device
        offloaded_devices_str = " and ".join(
            [device for device in set(device_map.values()) if device in ("cpu", "disk")]
        )
        if len(offloaded_devices_str) > 0:
            logger.warning(
                f"Some parameters are on the meta device because they were offloaded to the {offloaded_devices_str}."
            )

        # Attaching the hook may break tied weights, so we retie them
        retie_parameters(model, tied_params)

        # add warning to cuda and to method
        def add_warning(fn, model):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                warning_msg = "You shouldn't move a model that is dispatched using accelerate hooks."
                if str(fn.__name__) == "to":
                    to_device = torch._C._nn._parse_to(*args, **kwargs)[0]
                    if to_device is not None:
                        logger.warning(warning_msg)
                else:
                    logger.warning(warning_msg)
                for param in model.parameters():
                    if param.device == torch.device("meta"):
                        raise RuntimeError("You can't move a model that has some modules offloaded to cpu or disk.")
                return fn(*args, **kwargs)

            return wrapper

        # Make sure to update _accelerate_added_attributes in hooks.py if you add any hook
        model.to = add_warning(model.to, model)
        if is_npu_available():
            model.npu = add_warning(model.npu, model)
        elif is_mlu_available():
            model.mlu = add_warning(model.mlu, model)
        elif is_sdaa_available():
            model.sdaa = add_warning(model.sdaa, model)
        elif is_musa_available():
            model.musa = add_warning(model.musa, model)
        elif is_xpu_available():
            model.xpu = add_warning(model.xpu, model)
        else:
            model.cuda = add_warning(model.cuda, model)

        # Check if we are using multi-gpus with RTX 4000 series
        use_multi_gpu = len([device for device in set(device_map.values()) if device not in ("cpu", "disk")]) > 1
        if use_multi_gpu and not check_cuda_p2p_ib_support():
            logger.warning(
                "We've detected an older driver with an RTX 4000 series GPU. These drivers have issues with P2P. "
                "This can affect the multi-gpu inference when using accelerate device_map."
                "Please make sure to update your driver to the latest version which resolves this."
            )
    else:
        device = list(device_map.values())[0]
        # `torch.Tensor.to(<int num>)` is not supported by `torch_npu` (see this [issue](https://github.com/Ascend/pytorch/issues/16)).
        if is_npu_available() and isinstance(device, int):
            device = f"npu:{device}"
        elif is_mlu_available() and isinstance(device, int):
            device = f"mlu:{device}"
        elif is_sdaa_available() and isinstance(device, int):
            device = f"sdaa:{device}"
        elif is_musa_available() and isinstance(device, int):
            device = f"musa:{device}"
        if device != "disk":
            # Check if model has meta tensors and handle them properly
            has_meta_tensors = any(
                param.device.type == "meta" 
                for param in model.parameters()
            )
            
            if has_meta_tensors:
                # Use set_module_tensor_to_device for models with meta tensors
                from .utils.modeling import set_module_tensor_to_device
                for name, param in model.named_parameters():
                    if param.device.type == "meta":
                        set_module_tensor_to_device(model, name, device)
                for name, buffer in model.named_buffers():
                    if buffer.device.type == "meta":
                        set_module_tensor_to_device(model, name, device)
            else:
                # Use standard .to() for models without meta tensors
                model.to(device)
        else:
            raise ValueError(
                "You are trying to offload the whole model to the disk. Please use the `disk_offload` function instead."
            )
    # Convert OrderedDict back to dict for easier usage
    model.hf_device_map = dict(device_map)
    return model


def load_checkpoint_and_dispatch(
    model: nn.Module,
    checkpoint: Union[str, os.PathLike],
    device_map: Optional[Union[str, dict[str, Union[int, str, torch.device]]]] = None,
    max_memory: Optional[dict[Union[int, str], Union[int, str]]] = None,
    no_split_module_classes: Optional[list[str]] = None,
    offload_folder: Optional[Union[str, os.PathLike]] = None,
    offload_buffers: bool = False,
    dtype: Optional[Union[str, torch.dtype]] = None,
    offload_state_dict: Optional[bool] = None,
    skip_keys: Optional[Union[str, list[str]]] = None,
    preload_module_classes: Optional[list[str]] = None,
    force_hooks: bool = False,
    strict: bool = False,
    full_state_dict: bool = True,
    broadcast_from_rank0: bool = False,
):
    """
    Loads a (potentially sharded) checkpoint inside a model, potentially sending weights to a given device as they are
    loaded and adds the various hooks that will make this model run properly (even if split across devices).

    Args:
        model (`torch.nn.Module`): The model in which we want to load a checkpoint.
        checkpoint (`str` or `os.PathLike`):
            The folder checkpoint to load. It can be:
            - a path to a file containing a whole model state dict
            - a path to a `.json` file containing the index to a sharded checkpoint
            - a path to a folder containing a unique `.index.json` file and the shards of a checkpoint.
        device_map (`Dict[str, Union[int, str, torch.device]]`, *optional*):
            A map that specifies where each submodule should go. It doesn't need to be refined to each parameter/buffer
            name, once a given module name is inside, every submodule of it will be sent to the same device.

            To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For more
            information about each option see [here](../concept_guides/big_model_inference#designing-a-device-map).
            Defaults to None, which means [`dispatch_model`] will not be called.
        max_memory (`Dict`, *optional*):
            A dictionary device identifier to maximum memory. Will default to the maximum memory available for each GPU
            and the available CPU RAM if unset.
        no_split_module_classes (`List[str]`, *optional*):
            A list of layer class names that should never be split across device (for instance any layer that has a
            residual connection).
        offload_folder (`str` or `os.PathLike`, *optional*):
            If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            In the layers that are offloaded on the CPU or the hard drive, whether or not to offload the buffers as
            well as the parameters.
        dtype (`str` or `torch.dtype`, *optional*):
            If provided, the weights will be converted to that type when loaded.
        offload_state_dict (`bool`, *optional*):
            If `True`, will temporarily offload the CPU state dict on the hard drive to avoid getting out of CPU RAM if
            the weight of the CPU state dict + the biggest shard does not fit. Will default to `True` if the device map
            picked contains `"disk"` values.
        skip_keys (`str` or `List[str]`, *optional*):
            A list of keys to ignore when moving inputs or outputs between devices.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.
        force_hooks (`bool`, *optional*, defaults to `False`):
            Whether or not to force device hooks to be attached to the model even if all layers are dispatched to a
            single device.
        strict (`bool`, *optional*, defaults to `False`):
            Whether to strictly enforce that the keys in the checkpoint state_dict match the keys of the model's
            state_dict.
        full_state_dict (`bool`, *optional*, defaults to `True`): if this is set to `True`, all the tensors in the
            loaded state_dict will be gathered. No ShardedTensor and DTensor will be in the loaded state_dict.
        broadcast_from_rank0 (`False`, *optional*, defaults to `False`): when the option is `True`, a distributed
            `ProcessGroup` must be initialized. rank0 should receive a full state_dict and will broadcast the tensors
            in the state_dict one by one to other ranks. Other ranks will receive the tensors and shard (if applicable)
            according to the local shards in the model.

    Example:

    ```python
    >>> from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    >>> from huggingface_hub import hf_hub_download
    >>> from transformers import AutoConfig, AutoModelForCausalLM

    >>> # Download the Weights
    >>> checkpoint = "EleutherAI/gpt-j-6B"
    >>> weights_location = hf_hub_download(checkpoint, "pytorch_model.bin")

    >>> # Create a model and initialize it with empty weights
    >>> config = AutoConfig.from_pretrained(checkpoint)
    >>> with init_empty_weights():
    ...     model = AutoModelForCausalLM.from_config(config)

    >>> # Load the checkpoint and dispatch it to the right devices
    >>> model = load_checkpoint_and_dispatch(
    ...     model, weights_location, device_map="auto", no_split_module_classes=["GPTJBlock"]
    ... )
    ```
    """
    if isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
        raise ValueError(
            "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or 'sequential'."
        )
    if isinstance(device_map, str):
        if device_map != "sequential":
            max_memory = get_balanced_memory(
                model,
                max_memory=max_memory,
                no_split_module_classes=no_split_module_classes,
                dtype=dtype,
                low_zero=(device_map == "balanced_low_0"),
            )
        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=no_split_module_classes,
            dtype=dtype,
            offload_buffers=offload_buffers,
        )
    if offload_state_dict is None and device_map is not None and "disk" in device_map.values():
        offload_state_dict = True
    load_checkpoint_in_model(
        model,
        checkpoint,
        device_map=device_map,
        offload_folder=offload_folder,
        dtype=dtype,
        offload_state_dict=offload_state_dict,
        offload_buffers=offload_buffers,
        strict=strict,
        full_state_dict=full_state_dict,
        broadcast_from_rank0=broadcast_from_rank0,
    )
    if device_map is None:
        return model
    return dispatch_model(
        model,
        device_map=device_map,
        offload_dir=offload_folder,
        offload_buffers=offload_buffers,
        skip_keys=skip_keys,
        preload_module_classes=preload_module_classes,
        force_hooks=force_hooks,
    )


def attach_layerwise_casting_hooks(
    module: torch.nn.Module,
    storage_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    skip_modules_pattern: Union[str, tuple[str, ...]] = None,
    skip_modules_classes: Optional[tuple[type[torch.nn.Module], ...]] = None,
    non_blocking: bool = False,
) -> None:
    r"""
    Applies layerwise casting to a given module. The module expected here is a PyTorch `nn.Module`. This is helpful for
    reducing memory requirements when one doesn't want to fully quantize a model. Model params can be kept in say,
    `torch.float8_e4m3fn` and upcasted to a higher precision like `torch.bfloat16` during forward pass and downcasted
    back to `torch.float8_e4m3fn` to realize memory savings.

    Args:
        module (`torch.nn.Module`):
            The module whose leaf modules will be cast to a high precision dtype for computation, and to a low
            precision dtype for storage.
        storage_dtype (`torch.dtype`):
            The dtype to cast the module to before/after the forward pass for storage.
        compute_dtype (`torch.dtype`):
            The dtype to cast the module to during the forward pass for computation.
        skip_modules_pattern (`tuple[str, ...]`, defaults to `None`):
            A list of patterns to match the names of the modules to skip during the layerwise casting process. If set
            to `None` alongside `skip_modules_classes` being `None`, the layerwise casting is applied directly to the
            module instead of its internal submodules.
        skip_modules_classes (`tuple[type[torch.nn.Module], ...]`, defaults to `None`):
            A list of module classes to skip during the layerwise casting process.
        non_blocking (`bool`, defaults to `False`):
            If `True`, the weight casting operations are non-blocking.

    Example:

    ```python
    >>> from accelerate.hooks import attach_layerwise_casting_hooks
    >>> from transformers import AutoModelForCausalLM
    >>> import torch

    >>> # Model
    >>> checkpoint = "EleutherAI/gpt-j-6B"
    >>> model = AutoModelForCausalLM.from_pretrained(checkpoint)

    >>> # Attach hooks and perform inference
    >>> attach_layerwise_casting_hooks(model, storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    >>> with torch.no_grad():
    ...     model(...)
    ```

    Users can also pass modules they want to avoid from getting downcasted.

    ```py
    >>> attach_layerwise_casting_hooks(
    ...     model, storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16, skip_modules_pattern=["norm"]
    ... )
    ```
    """
    _attach_layerwise_casting_hooks(
        module, storage_dtype, compute_dtype, skip_modules_pattern, skip_modules_classes, non_blocking
    )


def _attach_layerwise_casting_hooks(
    module: torch.nn.Module,
    storage_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    skip_modules_pattern: Union[str, tuple[str, ...]] = None,
    skip_modules_classes: Optional[tuple[type[torch.nn.Module], ...]] = None,
    non_blocking: bool = False,
    _prefix: str = "",
):
    should_skip = (skip_modules_classes is not None and isinstance(module, skip_modules_classes)) or (
        skip_modules_pattern is not None and any(re.search(pattern, _prefix) for pattern in skip_modules_pattern)
    )
    if should_skip:
        logger.debug(f'Skipping layerwise casting for layer "{_prefix}"')
        return

    if isinstance(module, SUPPORTED_PYTORCH_LAYERS_FOR_UPCASTING):
        logger.debug(f'Applying layerwise casting to layer "{_prefix}"')
        add_hook_to_module(
            module,
            LayerwiseCastingHook(storage_dtype=storage_dtype, compute_dtype=compute_dtype, non_blocking=non_blocking),
            append=True,
        )
        return

    for name, submodule in module.named_children():
        layer_name = f"{_prefix}.{name}" if _prefix else name
        _attach_layerwise_casting_hooks(
            submodule,
            storage_dtype,
            compute_dtype,
            skip_modules_pattern,
            skip_modules_classes,
            non_blocking,
            _prefix=layer_name,
        )
