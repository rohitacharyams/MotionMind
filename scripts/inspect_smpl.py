"""Load SMPL pkl and convert chumpy arrays to numpy."""
import pickle, io, numpy as np

def load_smpl_model(path):
    """Load SMPL pkl converting chumpy Ch arrays to plain numpy arrays."""
    import types
    import sys

    # Create a fake chumpy module so pickle can resolve the class references
    chumpy_mod = types.ModuleType('chumpy')
    chumpy_ch_mod = types.ModuleType('chumpy.ch')
    chumpy_ch_ops_mod = types.ModuleType('chumpy.ch_ops')

    _pending = {}  # id -> numpy array, to resolve after unpickling

    class FakeChArray:
        """Standin for chumpy.Ch that stores data for later conversion."""
        def __init__(self, *args, **kwargs):
            self._data = None
        def __setstate__(self, state):
            if isinstance(state, dict) and 'x' in state:
                x = state['x']
                self._data = np.asarray(x) if hasattr(x, '__array__') else np.array(x)
            elif isinstance(state, dict):
                self._data = np.array(0.0)
            else:
                self._data = np.array(state)
        def __array__(self):
            return self._data if self._data is not None else np.array(0.0)

    chumpy_mod.Ch = FakeChArray
    chumpy_mod.array = FakeChArray
    chumpy_ch_mod.Ch = FakeChArray
    chumpy_ch_ops_mod.Ch = FakeChArray

    sys.modules['chumpy'] = chumpy_mod
    sys.modules['chumpy.ch'] = chumpy_ch_mod
    sys.modules['chumpy.ch_ops'] = chumpy_ch_ops_mod

    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    # Clean up
    del sys.modules['chumpy']
    del sys.modules['chumpy.ch']
    del sys.modules['chumpy.ch_ops']

    # Convert any remaining chumpy-like objects to numpy
    cleaned = {}
    for k, v in data.items():
        if isinstance(v, FakeChArray):
            cleaned[k] = np.array(v)
        elif hasattr(v, 'toarray'):  # sparse matrix
            cleaned[k] = v
        elif hasattr(v, '__array__'):
            cleaned[k] = np.asarray(v)
        else:
            cleaned[k] = v
    return cleaned


path = r'c:\dan\data\models\smpl_raw\smpl\models\basicModel_f_lbs_10_207_0_v1.0.0.pkl'
data = load_smpl_model(path)

print('Keys:', list(data.keys()))
for k, v in data.items():
    if hasattr(v, 'shape'):
        tp = type(v).__name__
        print(f'  {k}: {tp} shape={v.shape}')
    elif isinstance(v, (int, float, str)):
        print(f'  {k}: {v}')
    else:
        print(f'  {k}: {type(v).__name__}')

# Check key arrays
print('\n--- Key arrays ---')
print('v_template (mean shape):', data['v_template'].shape)
print('f (faces):', data['f'].shape)
print('J_regressor:', data['J_regressor'].shape)
print('weights:', data['weights'].shape)
print('kintree_table:', data['kintree_table'])
print('shapedirs:', data['shapedirs'].shape)
print('posedirs:', data['posedirs'].shape)
