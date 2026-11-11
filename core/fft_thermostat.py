from concurrent.futures import ThreadPoolExecutor
from numbers import Integral

import numpy as np
import scipy
from numba import njit, prange
from scipy.fft import irfft, next_fast_len, rfft

from .constants import BOLTZMANN_CONSTANT, HBAR

FFT_THERMOSTAT_SCHEME = 'sliding_fft_philox_thermal_bose_v1'
FFT_PARAMETER_DEFAULTS = {
    'quantum_fft_interval_fs': 0.0,
    'quantum_fft_half_width': 1024,
    'quantum_fft_block': 512,
    'quantum_fft_batch_channels': 768,
    'quantum_fft_workers': 1,
    'quantum_fft_dtype': 'float32',
}


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def bose_ratio(x):
    x = np.abs(np.asarray(x, dtype=np.float64))
    result = np.zeros_like(x)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        np.divide(x, np.expm1(x), out=result, where=x != 0)
    return np.where(x == 0, 1.0, result)


def design_filter(temperature, interval, half_width=1024):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('filter temperature must be finite and positive')
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError('filter interval must be finite and positive')
    half_width = _integer(half_width, 'half_width', 8)
    length = next_fast_len(max(65536, 32 * (2 * half_width + 1)))
    frequency = np.fft.rfftfreq(length, interval)
    theta = bose_ratio(HBAR * 2 * np.pi * frequency / (BOLTZMANN_CONSTANT * temperature))
    amplitude = np.sqrt(theta) / np.sinc(frequency * interval) ** 2
    impulse = irfft(amplitude, n=length)
    kernel = np.r_[impulse[-half_width:], impulse[:half_width + 1]]
    kernel *= np.kaiser(len(kernel), 8.6)
    return kernel


@njit(cache=True, inline='always')
def _interpolate_site(i, a, b, c, d, e, f, fractions, scale, start, middle, end):
    for j in range(3):
        start[i, j] = scale * ((1 - fractions[0]) * a[i, j] + fractions[0] * b[i, j])
        middle[i, j] = scale * ((1 - fractions[1]) * c[i, j] + fractions[1] * d[i, j])
        end[i, j] = scale * ((1 - fractions[2]) * e[i, j] + fractions[2] * f[i, j])


@njit(cache=True)
def _interpolate_serial(a, b, c, d, e, f, fractions, scale, start, middle, end):
    for i in range(start.shape[0]):
        _interpolate_site(i, a, b, c, d, e, f, fractions, scale, start, middle, end)


@njit(cache=True, parallel=True)
def _interpolate_parallel(a, b, c, d, e, f, fractions, scale, start, middle, end):
    for i in prange(start.shape[0]):
        _interpolate_site(i, a, b, c, d, e, f, fractions, scale, start, middle, end)


class SlidingFFTQuantumThermostat:

    def __init__(self, N, temperature, dt, gamma, moment, seed=None, *,
                 interval=None, half_width=1024, block=512, batch_channels=768,
                 workers=1, dtype='float32'):
        self.N = _integer(N, 'N')
        self.temperature = float(temperature)
        self.dt, self.gamma, self.moment = map(float, (dt, gamma, moment))
        if not np.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError('temperature must be finite and non-negative')
        if any(not np.isfinite(x) or x <= 0 for x in (self.dt, self.gamma, self.moment)):
            raise ValueError('dt, gamma and moment must be finite and positive')
        self._configured_seed = None if seed is None else _integer(seed, 'seed', 0)
        self.seed = (int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
                     if seed is None else self._configured_seed)
        self.half_width = _integer(half_width, 'half_width', 8)
        self.block = _integer(block, 'block')
        self.batch_channels = _integer(batch_channels, 'batch_channels')
        self.workers = _integer(workers, 'workers')
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise ValueError('dtype must be float32 or float64')
        if interval is None:
            target = min(100e-15, np.pi * HBAR / (18 * BOLTZMANN_CONSTANT * self.temperature)) if self.temperature else 100e-15
            ratio = max(1, int(np.floor(target / self.dt + 1e-12)))
            interval = ratio * self.dt
        self.interval = float(interval)
        if not np.isfinite(self.interval) or self.interval <= 0:
            raise ValueError('interval must be finite and positive')
        ratio = self.interval / self.dt
        if not np.isfinite(ratio) or not np.isclose(ratio, round(ratio), rtol=0, atol=1e-9):
            raise ValueError('FFT bath interval must be an integer multiple of spin dt')
        self.ratio = int(round(ratio))
        if self.ratio < 1:
            raise ValueError('FFT bath interval must be at least spin dt')
        self.kernel = design_filter(self.temperature, self.interval, self.half_width) if self.temperature else np.ones(1)
        self.nfft = next_fast_len(len(self.kernel) - 1 + self.block)
        self.kernel_fft = rfft(self.kernel, n=self.nfft)
        self.fine_step = 0
        self._cache = {}

    @classmethod
    def from_simulation(cls, simulation):
        p = dict(FFT_PARAMETER_DEFAULTS)
        p.update({k: v for k, v in simulation.params.items() if k in p})
        interval = p['quantum_fft_interval_fs']
        return cls(simulation.N, simulation.T, simulation.dt, simulation.gamma,
                   simulation.magnetic_moment, simulation.random_seed,
                   interval=None if interval is None or float(interval) == 0 else float(interval) * 1e-15,
                   half_width=p['quantum_fft_half_width'], block=p['quantum_fft_block'],
                   batch_channels=p['quantum_fft_batch_channels'], workers=p['quantum_fft_workers'],
                   dtype=p['quantum_fft_dtype'])

    @property
    def cache_capacity_bytes(self):
        return 2 * self.block * self.N * 3 * self.dtype.itemsize

    def state_dict(self):
        return dict(scheme=FFT_THERMOSTAT_SCHEME, N=self.N, temperature_K=self.temperature,
                    dt=self.dt, gamma=self.gamma, moment=self.moment, seed=self.seed,
                    interval=self.interval, half_width=self.half_width, block=self.block,
                    batch_channels=self.batch_channels, dtype=self.dtype.name,
                    fine_step=self.fine_step, numpy_version=np.__version__, scipy_version=scipy.__version__)

    def load_state_dict(self, state):
        state = dict(state)
        step = _integer(state.pop('fine_step'), 'fine_step', 0)
        seed = _integer(state.get('seed'), 'seed', 0)
        expected = self.state_dict()
        expected.pop('fine_step')
        if self._configured_seed is None:
            expected['seed'] = seed
        if state != expected:
            raise ValueError('FFT thermostat checkpoint configuration or library versions differ')
        self.seed = seed
        self.fine_step = step
        self._cache.clear()

    def _white(self, start, stop, first_channel, last_channel):
        width = last_channel - first_channel
        values = np.empty((width, stop - start))
        for segment in range(start // self.block, (stop - 1) // self.block + 1):
            signed = 2 * segment if segment >= 0 else -2 * segment - 1
            rng = np.random.Generator(np.random.Philox(np.random.SeedSequence(
                [self.seed, signed, first_channel, self.N, self.batch_channels])))
            white = rng.standard_normal((width, self.block))
            lo, hi = max(start, segment * self.block), min(stop, (segment + 1) * self.block)
            values[:, lo - start:hi - start] = white[:, lo - segment * self.block:hi - segment * self.block]
        return values

    def _refill(self, index):
        if len(self._cache) >= 2:
            del self._cache[min(self._cache)]
        output = np.empty((self.block, 3 * self.N), dtype=self.dtype)
        start = index * self.block - (len(self.kernel) - 1)
        stop = (index + 1) * self.block

        def calculate(first):
            last = min(first + self.batch_channels, 3 * self.N)
            white = self._white(start, stop, first, last)
            transformed = rfft(white, n=self.nfft, axis=1, workers=1)
            transformed *= self.kernel_fft[None, :]
            filtered = irfft(transformed, n=self.nfft, axis=1, workers=1)
            output[:, first:last] = filtered[:, len(self.kernel) - 1:len(self.kernel) - 1 + self.block].T

        batches = range(0, 3 * self.N, self.batch_channels)
        if self.workers == 1:
            for first in batches:
                calculate(first)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                for _ in executor.map(calculate, batches):
                    pass
        self._cache[index] = output.reshape(self.block, self.N, 3)

    def coarse_value(self, index):
        index = _integer(index, 'coarse index', 0)
        block, offset = divmod(index, self.block)
        if block not in self._cache:
            self._refill(block)
        return self._cache[block][offset]

    def fields(self, alpha, start, midpoint, end):
        alpha = float(alpha)
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError('damping must be finite and non-negative')
        outputs = (start, midpoint, end)
        if any(o.shape != (self.N, 3) or o.dtype != np.float64 for o in outputs):
            raise ValueError('field buffers must have shape (N, 3) and dtype float64')
        if self.temperature == 0 or alpha == 0:
            for out in outputs:
                out.fill(0)
            return
        pairs, fractions = [], []
        for stage in range(3):
            index, rem = divmod(2 * self.fine_step + stage, 2 * self.ratio)
            pairs.extend((self.coarse_value(index), self.coarse_value(index + 1)))
            fractions.append(rem / (2 * self.ratio))
        scale = np.sqrt(2 * alpha * BOLTZMANN_CONSTANT * self.temperature /
                        (self.gamma * self.moment * self.interval))
        interpolate = _interpolate_parallel if self.N >= 8192 else _interpolate_serial
        interpolate(*pairs, np.asarray(fractions), scale, *outputs)

    def advance(self):
        self.fine_step += 1


def add_fft_arguments(parser, *, sweep=False):
    nargs = {'nargs': '+'} if sweep else {}
    for name, default in FFT_PARAMETER_DEFAULTS.items():
        option = '--' + name.replace('_', '-')
        kind = float if name.endswith('_fs') else str if name.endswith('_dtype') else int
        extra = {'choices': ('float32', 'float64')} if name.endswith('_dtype') else {}
        parser.add_argument(option, type=kind, default=[default] if sweep else default,
                            **nargs, **extra)


def fft_parameters(options, *, sweep=False):
    return {k: np.asarray(getattr(options, k)) if sweep else getattr(options, k)
            for k in FFT_PARAMETER_DEFAULTS}
