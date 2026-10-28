import abc
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from unittest.mock import MagicMock


class MockAutogradFunction:
    @staticmethod
    def apply(*args, **kwargs):
        # If the first arg is a tensor, return a similar tensor
        if args and hasattr(args[0], "shape"):
            res = MockTensor(args[0].shape)
            res._val = 1.0  # Default to 1.0 for spiking tests
            res.requires_grad = any(getattr(a, "requires_grad", False) for a in args)
            res._grad_parents = list(args)
            return res
        return MockTensor()


class MockDevice:
    def __init__(self, type="cpu", index=None):
        self.type = type
        self.index = index

    def __repr__(self):
        return f"device(type='{self.type}', index={self.index})"

    def __eq__(self, other):
        if isinstance(other, str):
            return self.type == other
        if isinstance(other, MockDevice):
            return self.type == other.type and self.index == other.index
        return False


class MockTensor:
    def __init__(self, *args, **kwargs):
        self.device = MockDevice()
        self.dtype = MagicMock()

        # Handle shape from randn(1, 16, 16, 1) or randn((1, 16, 16, 1))
        if args:
            if isinstance(args[0], (list, tuple)):
                self.shape = tuple(args[0])
            else:
                self.shape = tuple(args)
        else:
            self.shape = (1, 10)
        self.data = self
        self.requires_grad = kwargs.get("requires_grad", False)
        self.grad = None
        self._grad_parents = []

    @property
    def real(self):
        return self

    @property
    def imag(self):
        return self

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return MagicMock()

    def __call__(self, *args, **kwargs):
        return MagicMock()

    def to(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def cuda(self):
        return self

    def detach(self):
        return self

    def unsqueeze(self, dim):
        new_shape = list(self.shape)
        if dim < 0:
            dim += len(new_shape) + 1
        new_shape.insert(dim, 1)
        return MockTensor(new_shape)

    def squeeze(self, dim=None):
        if dim is None:
            new_shape = [s for s in self.shape if s != 1]
            if not new_shape:
                new_shape = [1]
        else:
            new_shape = list(self.shape)
            if dim < 0:
                dim += len(new_shape)
            if new_shape[dim] == 1:
                new_shape.pop(dim)
        return MockTensor(new_shape)

    def permute(self, *args, **kwargs):
        return self

    def float(self, *args, **kwargs):
        return self

    def mean(self, *args, **kwargs):
        return self

    def max(self, *args, **kwargs):
        return self

    def min(self, *args, **kwargs):
        return self

    def flatten(self, *args, **kwargs):
        return self

    def abs(self):
        res = MockTensor(self.shape)
        res.requires_grad = self.requires_grad
        res._grad_parents = [self]
        if hasattr(self, "_val"):
            res._val = abs(self._val)
        return res

    def numel(self):
        return getattr(self, "_numel", 100)

    def __gt__(self, other):
        res = MockTensor((1,), _val=1.1)
        res._bool_val = True
        return res

    def __lt__(self, other):
        res = MockTensor((1,), _val=0.0)
        res._bool_val = False
        return res

    def __ge__(self, other):
        res = MockTensor((1,), _val=1.1)
        res._bool_val = True
        return res

    def __le__(self, other):
        res = MockTensor((1,), _val=0.0)
        res._bool_val = False
        return res

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            res = MockTensor((1,), _val=1.1)
            res._bool_val = True
            return res
        return MockTensor(self.shape)

    def __and__(self, other):
        return self

    def __or__(self, other):
        return self

    def __neg__(self):
        return self

    def __bool__(self):
        return getattr(self, "_bool_val", True)

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        new_shape = list(self.shape)
        out_shape = []
        for i, s in enumerate(idx):
            if i >= len(new_shape):
                break
            if isinstance(s, int):
                pass  # Dimension removed
            elif isinstance(s, slice):
                if s.start is None and s.stop is None and s.step is None:
                    out_shape.append(new_shape[i])
                elif s.step == 2:
                    out_shape.append(new_shape[i] // 2)
                elif s.stop is not None and s.start is None:
                    out_shape.append(min(new_shape[i], s.stop))
                elif s.start is not None and s.stop is not None:
                    out_shape.append(min(new_shape[i], s.stop - s.start))
                else:
                    out_shape.append(new_shape[i])
            else:
                out_shape.append(new_shape[i])
        if len(idx) < len(new_shape):
            out_shape.extend(new_shape[len(idx) :])
        if any(isinstance(x, (MockTensor, MagicMock)) for x in idx):
            return MockTensor(self.shape)
        return MockTensor(out_shape)

    def __setitem__(self, idx, val):
        pass  # Mock assignment

    def dim(self):
        return len(self.shape)

    def _op(self, other):
        res = MockTensor(self.shape)
        res.requires_grad = self.requires_grad or getattr(other, "requires_grad", False)
        res._grad_parents = [self, other]
        return res

    def __pow__(self, other):
        return self._op(other)

    def __add__(self, other):
        return self._op(other)

    def __radd__(self, other):
        return self._op(other)

    def __sub__(self, other):
        return self._op(other)

    def __rsub__(self, other):
        return self._op(other)

    def __mul__(self, other):
        return self._op(other)

    def __rmul__(self, other):
        return self._op(other)

    def __truediv__(self, other):
        return self._op(other)

    def __rtruediv__(self, other):
        return self._op(other)

    def item(self):
        if hasattr(self, "_val"):
            return self._val
        return 1.1

    def backward(self, *args, **kwargs):
        # Iterative backward simulation to avoid RecursionError
        stack = [self]
        visited = set()
        while stack:
            curr = stack.pop()
            if id(curr) in visited:
                continue
            visited.add(id(curr))

            parents = getattr(curr, "_grad_parents", [])
            for p in parents:
                if isinstance(p, MockTensor) and p.requires_grad:
                    if p.grad is None:
                        p.grad = MockTensor(p.shape)
                        p.grad._val = getattr(curr, "_val", 0.1)
                    stack.append(p)
                elif isinstance(p, (list, tuple, LazyMockModule)):
                    for item in p:
                        if isinstance(item, MockTensor) and item.requires_grad:
                            if item.grad is None:
                                item.grad = MockTensor(item.shape)
                                item.grad._val = getattr(curr, "_val", 0.1)
                            stack.append(item)
                        elif hasattr(item, "_grad_parents"):
                            stack.append(item)

    def diag(self):
        res = MockTensor((self.shape[0],))
        res.requires_grad = self.requires_grad
        res._grad_parents = [self]
        return res

    def sum(self, *args, **kwargs):
        # Handle dim argument
        dim = kwargs.get("dim", args[0] if args else None)
        if dim is not None:
            if isinstance(dim, int):
                dim = [dim]
            # Normalize dims
            ndims = len(self.shape)
            dim = [(d + ndims if d < 0 else d) for d in dim]
            new_shape = [s for i, s in enumerate(self.shape) if i not in dim]
            if not new_shape:
                new_shape = [1]
            res = MockTensor(new_shape)
        else:
            res = MockTensor((1,))
        res.requires_grad = self.requires_grad
        res._grad_parents = [self]
        res._val = getattr(self, "_val", 1.1)
        return res

    def dim(self):
        return len(self.shape)

    def size(self, dim=None):
        if dim is not None:
            return self.shape[dim]
        return self.shape

    @property
    def real(self):
        return self

    @property
    def imag(self):
        res = MockTensor(self.shape)
        res._is_complex = False
        return res

    def clone(self):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def cuda(self, *args, **kwargs):
        return self

    def to(self, *args, **kwargs):
        return self

    def type(self, *args, **kwargs):
        return self

    def float(self):
        return self

    def long(self):
        return self

    def half(self):
        return self

    def double(self):
        return self

    def bool(self):
        return True

    def numpy(self):
        return np.zeros(self.shape)

    def contiguous(self):
        return self

    def repeat(self, *args):
        new_shape = list(self.shape)
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        for i, r in enumerate(args):
            if i < len(new_shape):
                new_shape[i] *= r
            else:
                new_shape.insert(0, r)
        return MockTensor(new_shape)

    def __len__(self):
        return self.shape[0] if self.shape else 0

    def __getitem__(self, idx):
        # Handle slicing/indexing by returning a tensor with modified shape
        if isinstance(idx, (tuple, list)):
            new_shape = list(self.shape)
            # This is a very crude approximation of slicing
            for i, s in enumerate(idx):
                if i < len(new_shape) and isinstance(s, int):
                    # Reduction (actually it should be removed or set to 1)
                    # For simplicity, we keep it as 1 to avoid complexity of real slicing
                    new_shape[i] = 1
            res = MockTensor([s for s in new_shape if s > 0])
        else:
            res = MockTensor(self.shape[1:] if len(self.shape) > 1 else (1,))
        res.requires_grad = self.requires_grad
        res._grad_parents = [self]
        return res


class MockBaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def dict(self):
        return self.__dict__

    def model_dump(self):
        return self.__dict__


class TrainingConfigSchema(MockBaseModel):
    pass


class CheckpointConfigSchema(MockBaseModel):
    pass


class MixedPrecisionConfig(MockBaseModel):
    pass


class AugmentationConfigSchema(MockBaseModel):
    pass


# TrainingSettings for spectramr.config.settings
class TrainingSettings(MockBaseModel):
    """Mock TrainingSettings class to support isinstance checks."""

    @classmethod
    def from_yaml(cls, path):
        return cls(
            training=MockBaseModel(
                training_mode="reconstruction",
                epochs=100,
                batch_size=4,
                gradient_accumulation_steps=1,
            ),
            optimization=MockBaseModel(
                optimizer="adam",
                learning_rate=1e-4,
                scheduler=MockBaseModel(type="cosine", kwargs={}),
                use_amp=False,
                weight_decay=0.0,
                beta1=0.9,
                beta2=0.999,
            ),
            model=MockBaseModel(model_type="unet"),
            logging=MockBaseModel(log_dir="logs"),
            checkpoint=MockBaseModel(save_interval=1000),
        )


class MockLayer:
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if len(args) > 1 and isinstance(args[1], int):
            self.out_channels = args[1]
        elif "out_channels" in kwargs:
            self.out_channels = kwargs["out_channels"]
        elif "out_features" in kwargs:
            self.out_channels = kwargs["out_features"]

        if hasattr(self, "out_channels"):
            self.weight = MockTensor((self.out_channels, 1), requires_grad=True)
            self.bias = MockTensor((self.out_channels,), requires_grad=True)

    def __call__(self, *args, **kwargs):
        if args and hasattr(args[0], "shape"):
            out_shape = list(args[0].shape)
            if hasattr(self, "out_channels") and len(out_shape) >= 2:
                out_shape[1] = self.out_channels
            res = MockTensor(out_shape)
            # Propagate requires_grad
            res.requires_grad = any(
                getattr(a, "requires_grad", False) for a in args
            ) or any(getattr(p, "requires_grad", False) for p in self.parameters())
            # Propagate gradients through layers
            res._grad_parents = list(args) + self.parameters()
            return res
        return MockTensor()

    def parameters(self, recurse=True):
        params = []
        for v in self.__dict__.values():
            if isinstance(v, MockTensor) and v.requires_grad:
                params.append(v)
        return params

    def named_parameters(self, prefix="", recurse=True):
        named_params = []
        for v in self.__dict__.keys():
            val = getattr(self, v)
            if isinstance(val, MockTensor) and val.requires_grad:
                name = f"{prefix}.{v}" if prefix else v
                named_params.append((name, val))
        return named_params

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def train(self, mode=True):
        return self


class LazyMockModule(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__spec__ = None
        self.__path__ = []
        # Store items if this is used as a container (e.g., ModuleList or Sequential)
        if args and isinstance(args[0], (list, tuple)):
            self._mock_items = args[0]
        else:
            self._mock_items = list(args)

    def __getattr__(self, name):
        if name in ("__path__", "__file__", "__spec__"):
            return None
        child = super().__getattr__(name)
        if isinstance(child, MagicMock) and not isinstance(child, LazyMockModule):
            object.__setattr__(self, name, LazyMockModule())
            return getattr(self, name)
        return child

    def parameters(self, recurse=True):
        res = []
        for item in self._mock_items:
            if hasattr(item, "parameters"):
                res.extend(item.parameters(recurse))
        return res

    def named_parameters(self, prefix="", recurse=True):
        res = []
        for i, item in enumerate(self._mock_items):
            p = f"{prefix}.{i}" if prefix else str(i)
            if hasattr(item, "named_parameters"):
                res.extend(item.named_parameters(prefix=p, recurse=recurse))
        return res

    def __iter__(self):
        return iter(self._mock_items)

    def __len__(self):
        return len(self._mock_items)

    def __getitem__(self, idx):
        return self._mock_items[idx]


try:
    from spectramr.shared.utils.metaclass import ModuleABCMeta
except ImportError:

    class ModuleABCMeta(abc.ABCMeta):
        pass


try:
    from typing import _ProtocolMeta
except ImportError:

    class _ProtocolMeta(type):
        pass


class MockModuleMeta(ModuleABCMeta, _ProtocolMeta):
    pass


class MockModule(metaclass=MockModuleMeta):
    def __init__(self, *args, **kwargs):
        self._parameters = {}
        self._buffers = {}
        self._modules = {}
        self.training = True

    def __call__(self, *args, **kwargs):
        if hasattr(self, "forward"):
            res = self.forward(*args, **kwargs)
            if isinstance(res, MockTensor):
                res._grad_parents = list(args) + self.parameters()
            return res
        return MockTensor()

    def parameters(self, recurse=True):
        params = []
        for v in self.__dict__.values():
            if isinstance(v, MockTensor):
                params.append(v)
            elif isinstance(v, (MockModule, LazyMockModule, MockLayer)):
                if hasattr(v, "parameters"):
                    params.extend(v.parameters(recurse))
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if hasattr(item, "parameters"):
                        params.extend(item.parameters(recurse))
        return params

    def named_parameters(self, prefix="", recurse=True):
        named_params = []
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            name = f"{prefix}.{k}" if prefix else k
            if isinstance(v, MockTensor):
                named_params.append((name, v))
            elif isinstance(v, (MockModule, LazyMockModule, MockLayer)):
                if hasattr(v, "named_parameters"):
                    named_params.extend(
                        v.named_parameters(prefix=name, recurse=recurse)
                    )
            elif isinstance(v, (list, tuple)):
                for i, item in enumerate(v):
                    if hasattr(item, "named_parameters"):
                        named_params.extend(
                            item.named_parameters(prefix=f"{name}.{i}", recurse=recurse)
                        )
        return named_params

    def to(self, *args, **kwargs):
        return self

    def cuda(self, device=None):
        return self

    def cpu(self):
        return self

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict, strict=True):
        return MagicMock()

    def add_module(self, name, module):
        self._modules[name] = module

    def register_parameter(self, name, param):
        self._parameters[name] = param

    def register_buffer(self, name, buffer, persistent=True):
        self._buffers[name] = buffer


class MockLoader(Loader):
    def create_module(self, spec):
        if spec.name in sys.modules:
            return sys.modules[spec.name]
        if spec.name == "torch" or spec.name.startswith("torch."):
            torch = setup_mock_torch_instance()
            if spec.name in sys.modules:
                return sys.modules[spec.name]
            return torch
        if spec.name == "yaml":
            return setup_mock_yaml_instance()
        if spec.name == "torchio":
            return setup_mock_torchio_instance()
        if spec.name == "pydantic":
            return setup_mock_pydantic_instance()
        if spec.name == "pydantic_settings":
            return setup_mock_pydantic_settings_instance()
        parent_name = spec.name.rsplit(".", 1)[0]
        if parent_name in sys.modules:
            parent = sys.modules[parent_name]
            attr_name = spec.name.rsplit(".", 1)[1]
            if hasattr(parent, attr_name):
                return getattr(parent, attr_name)
        return LazyMockModule()

    def exec_module(self, module):
        pass


class MockFinder(MetaPathFinder):
    def __init__(self, prefixes, exclusions=None):
        self.prefixes = prefixes
        self.exclusions = exclusions or []

    def find_spec(self, fullname, path, target=None):
        if any(fullname == e or fullname.startswith(e + ".") for e in self.exclusions):
            return None
        if any(fullname == p or fullname.startswith(p + ".") for p in self.prefixes):
            return ModuleSpec(fullname, MockLoader())
        return None


def setup_mock_torch_instance():
    if "torch" in sys.modules:
        return sys.modules["torch"]
    torch = LazyMockModule()
    nn = torch.nn
    utils = torch.utils
    data = torch.utils.data
    optim = torch.optim
    torch.nn.Module = MockModule
    functional = LazyMockModule()
    fft = LazyMockModule()
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn
    sys.modules["torch.nn.functional"] = functional
    sys.modules["torch.fft"] = fft
    sys.modules["torch.utils"] = utils
    sys.modules["torch.utils.data"] = data
    sys.modules["torch.optim"] = optim
    torch.nn.functional = functional
    torch.fft = fft
    torch.autograd = LazyMockModule()
    torch.autograd.Function = MockAutogradFunction

    def mock_grad(outputs, inputs, **kwargs):
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        for o in outputs:
            if hasattr(o, "backward"):
                o.backward()
        res = []
        for i in inputs:
            g = getattr(i, "grad", None)
            if g is None:
                g = MockTensor(getattr(i, "shape", (1,)))
            res.append(g)
        return res

    torch.autograd.grad = mock_grad
    sys.modules["torch.autograd"] = torch.autograd
    # Pydantic and Config mocks
    setup_universal_mock(["pydantic", "pydantic_settings"])

    # Inject TrainingSettings into sys.modules to ensure isinstance works
    if "spectramr.config.settings" not in sys.modules:
        settings_module = LazyMockModule()
        settings_module.TrainingSettings = TrainingSettings
        sys.modules["spectramr.config.settings"] = settings_module
    elif not hasattr(sys.modules["spectramr.config.settings"], "TrainingSettings"):
        sys.modules["spectramr.config.settings"].TrainingSettings = TrainingSettings

    # Inject ModuleABCMeta to resolve metaclass conflicts
    if "spectramr.shared.utils.metaclass" not in sys.modules:
        meta_module = LazyMockModule()
        meta_module.ModuleABCMeta = MockModuleMeta
        sys.modules["spectramr.shared.utils.metaclass"] = meta_module
    else:
        sys.modules["spectramr.shared.utils.metaclass"].ModuleABCMeta = MockModuleMeta

    torch.nn.Linear = MockLayer
    torch.nn.Conv2d = MockLayer
    torch.nn.ConvTranspose2d = MockLayer
    torch.nn.ReLU = MockLayer
    torch.nn.Sigmoid = MockLayer
    torch.nn.Tanh = MockLayer
    torch.nn.Flatten = MockLayer
    torch.nn.Sequential = MockLayer
    torch.nn.Upsample = MockLayer
    torch.nn.BatchNorm2d = MockLayer
    torch.nn.MaxPool2d = MockLayer
    torch.nn.L1Loss = MockLayer
    torch.nn.MSELoss = MockLayer
    torch.nn.CrossEntropyLoss = MockLayer
    torch.nn.BCEWithLogitsLoss = MockLayer
    torch.nn.KLDivLoss = MockLayer
    torch.nn.Parameter = lambda data=None, requires_grad=True: (
        data if data is not None else MockTensor()
    )
    torch.device = MockDevice

    class MockOptimizer(MagicMock):
        def __init__(self, params=None, defaults=None, **kwargs):
            super().__init__()
            self.defaults = defaults or {}
            self.defaults.update(kwargs)
            if params is not None:
                self.param_groups = [{"params": params, **self.defaults}]
            else:
                self.param_groups = []

    class Adam(MockOptimizer):
        pass

    class AdamW(MockOptimizer):
        pass

    class SGD(MockOptimizer):
        pass

    class RMSprop(MockOptimizer):
        pass

    torch.optim.Optimizer = MockOptimizer
    torch.optim.Adam = Adam
    torch.optim.AdamW = AdamW
    torch.optim.SGD = SGD
    torch.optim.RMSprop = RMSprop

    torch.Tensor = MockTensor
    torch.randn = lambda *args, **kwargs: MockTensor(*args, **kwargs)
    torch.zeros = lambda *args, **kwargs: MockTensor(*args, **kwargs)
    torch.ones = lambda *args, **kwargs: MockTensor(*args, **kwargs)
    torch.empty = lambda *args, **kwargs: MockTensor(*args, **kwargs)
    torch.linspace = lambda start, end, steps, **kwargs: MockTensor((steps,))
    torch.arange = lambda *args, **kwargs: MockTensor(
        (len(range(*args)) if args and isinstance(args[0], int) else 10,)
    )
    torch.full = lambda *args, **kwargs: MockTensor(
        args[0] if len(args) > 0 else (1, 10), **kwargs
    )

    def mock_complex(real, imag, **kwargs):
        res = MockTensor(real.shape)
        res._is_complex = True
        return res

    torch.complex = mock_complex
    torch.is_complex = lambda x: getattr(x, "_is_complex", False)

    def mock_cat(tensors, dim=0):
        if not tensors:
            return MockTensor()
        shapes = [t.shape for t in tensors]
        out_shape = list(shapes[0])
        out_shape[dim] = sum(s[dim] for s in shapes)
        req_grad = any(getattr(t, "requires_grad", False) for t in tensors)
        return MockTensor(out_shape, requires_grad=req_grad)

    torch.cat = mock_cat

    def mock_tensor_init(*args, **kwargs):
        t = MockTensor(*args, **kwargs)
        if args and hasattr(args[0], "__len__") and len(args[0]) == 0:
            t._numel = 0
            t.shape = (0,)
        if args and isinstance(args[0], (list, tuple)) and len(args[0]) == 1:
            try:
                t._val = float(args[0][0])
            except:
                pass
        return t

    torch.tensor = mock_tensor_init
    torch.from_numpy = lambda x: MockTensor(x.shape)
    torch.sqrt = lambda x: x
    torch.exp = lambda x: x
    torch.abs = lambda x: x
    torch.softmax = lambda x, dim=None: x
    torch.sigmoid = lambda x: x

    def mock_stack(tensors, dim=0):
        if not tensors:
            return MockTensor()
        shape = list(tensors[0].shape)
        shape.insert(dim, len(tensors))
        return MockTensor(shape)

    torch.stack = mock_stack
    torch.meshgrid = lambda *args, **kwargs: (MockTensor(), MockTensor())
    torch.nn.functional.max_pool2d = lambda x, *args, **kwargs: x
    torch.nn.functional.max_pool3d = lambda x, *args, **kwargs: x
    torch.nn.functional.sigmoid = lambda x: x
    torch.nn.functional.relu = lambda x, inplace=False: x
    torch.nn.functional.pad = lambda x, pad, mode="constant", value=0: x
    torch.nn.functional.cross_entropy = lambda input, target, **kwargs: input.sum()
    torch.nn.functional.mse_loss = lambda input, target, **kwargs: input.sum()
    torch.nn.functional.l1_loss = lambda input, target, **kwargs: input.sum()
    torch.nn.functional.conv2d = (
        lambda x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1: (
            MockTensor(x.shape)
        )
    )
    torch.nn.functional.conv_transpose2d = lambda x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1: (
        MockTensor(x.shape)
    )

    def mock_interpolate(x, size=None, scale_factor=None, **kwargs):
        new_shape = list(x.shape)
        if size is not None:
            if isinstance(size, int):
                size = (size,)
            new_shape[-len(size) :] = list(size)
        elif scale_factor is not None:
            if isinstance(scale_factor, (float, int)):
                scale_factor = [scale_factor] * 2
            # Very rough approx
            for i in range(len(scale_factor)):
                new_shape[-1 - i] = int(new_shape[-1 - i] * scale_factor[-1 - i])
        return MockTensor(new_shape)

    torch.nn.functional.interpolate = mock_interpolate

    def mock_grid_sample(input, grid, **kwargs):
        return MockTensor(input.shape)

    torch.nn.functional.grid_sample = mock_grid_sample

    def mock_affine_grid(theta, size, **kwargs):
        # size is [B, C, D, H, W] or [B, C, H, W]
        # output is [B, D, H, W, 3] or [B, H, W, 2]
        res_shape = list(size)
        res_shape.pop(1)  # Remove C
        res_shape.append(len(size) - 2)  # Add coordinate dim
        return MockTensor(res_shape)

    torch.nn.functional.affine_grid = mock_affine_grid
    torch.fft.fft2 = lambda x, **kwargs: x
    torch.fft.ifft2 = lambda x, **kwargs: x
    torch.fft.fftn = lambda x, **kwargs: x
    torch.fft.ifftn = lambda x, **kwargs: x
    torch.fft.ifftshift = lambda x, **kwargs: x
    torch.fft.fftshift = lambda x, **kwargs: x

    class MockNoGrad:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args, **kwargs):
            pass

        def __call__(self, arg):
            return arg if callable(arg) else self

    torch.no_grad = MockNoGrad
    torch.cuda.memory_allocated = lambda *args, **kwargs: 0
    torch.cuda.empty_cache = lambda: None
    torch.cuda.is_available = lambda: False  # Default to False to simplify tests
    return torch


class MockBaseModel:
    model_config = {}

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_yaml(cls, path):
        return cls()


def setup_mock_yaml_instance():
    yaml = LazyMockModule()
    yaml.YAMLError = Exception
    return yaml


def setup_mock_torchio_instance():
    tio = LazyMockModule()

    class MockTransform:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, subject):
            if hasattr(self, "apply_transform"):
                return self.apply_transform(subject)
            return subject

    class MockSubject(dict):
        def __init__(self, **kwargs):
            super().__init__(kwargs)
            for k, v in kwargs.items():
                setattr(self, k, v)

        def get_images_names(self):
            return [k for k, v in self.items() if hasattr(v, "affine")]

        def get_images(self):
            return [v for k, v in self.items() if hasattr(v, "affine")]

    class MockImage:
        def __init__(self, tensor=None, affine=None, **kwargs):
            self.data = tensor
            self.affine = affine
            self.shape = (1, 1, 1, 1)
            if tensor is not None and hasattr(tensor, "shape"):
                self.shape = tensor.shape
            for k, v in kwargs.items():
                setattr(self, k, v)

    tio.Transform = MockTransform
    tio.transforms = LazyMockModule()
    tio.transforms.Transform = MockTransform
    tio.Subject = MockSubject
    tio.ScalarImage = MockImage
    tio.LabelMap = MockImage
    tio.ZNormalization = MockTransform
    tio.Compose = MockTransform
    tio.Resize = MockTransform
    tio.CropOrPad = MockTransform
    tio.data.UniformSampler = MagicMock
    tio.Queue = MagicMock
    return tio


def setup_mock_pydantic_settings_instance():
    ps = LazyMockModule()
    ps.BaseSettings = MockBaseModel
    ps.SettingsConfigDict = lambda *args, **kwargs: {}
    return ps


def setup_mock_pydantic_instance():
    pydantic = LazyMockModule()
    pydantic.BaseModel = MockBaseModel
    pydantic.TrainingConfigSchema = TrainingConfigSchema
    pydantic.CheckpointConfigSchema = CheckpointConfigSchema
    pydantic.AugmentationConfigSchema = AugmentationConfigSchema
    pydantic.Field = lambda *args, **kwargs: MagicMock()

    # pydantic_settings support
    pydantic.BaseSettings = MockBaseModel
    pydantic.SettingsConfigDict = lambda *args, **kwargs: {}
    return pydantic


def setup_universal_mock(prefixes=None, exclusions=None):
    if prefixes is None:
        prefixes = [
            "torch",
            "yaml",
            "h5py",
            "nibabel",
            "torchio",
            "torchkbnufft",
            "mamba_ssm",
            "einops",
            "cv2",
            "numpy",
            "scipy",
            "sklearn",
            "matplotlib",
            "pandas",
            "PIL",
            "seaborn",
            "pydantic",
            "pydantic_settings",
            "torchvision",
            "optuna",
            "jaxtyping",
            "kornia",
            "clearml",
            "wandb",
            "medpy",
            "psutil",
        ]
    existing_finder = next(
        (f for f in sys.meta_path if isinstance(f, MockFinder)), None
    )
    if existing_finder:
        for p in prefixes:
            if p not in existing_finder.prefixes:
                existing_finder.prefixes.append(p)
        if exclusions:
            for e in exclusions:
                if e not in existing_finder.exclusions:
                    existing_finder.exclusions.append(e)
        return
    sys.meta_path.insert(0, MockFinder(prefixes, exclusions))


def setup_mock_torch():
    setup_universal_mock(["torch"])


def setup_mock_yaml():
    setup_universal_mock(["yaml"])


def setup_mock_heavy_dependencies():
    setup_universal_mock()
