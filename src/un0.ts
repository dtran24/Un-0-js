// Un-0 (all six released checkpoints: CIFAR-10 32×32 fp32, ImageNet-64 64×64
// fp16) forward pass in jax-js, on the WebGPU backend when available and Wasm
// otherwise. Mirrors un0/model.py in eval mode: ConditionalKuramotoDynamics
// -> fixed-step Euler -> relativization (mean_relative or ref_oscillator, per
// checkpoint) + sin_cos readout -> ResizeConvDecoder (convs with biases).
// Model shapes are read from each checkpoint's own safetensors metadata.
// Weights: https://huggingface.co/un-ai/Un-0 (MIT), converted to safetensors
// with the K / K_cond diagonals pre-zeroed and mup scales baked in.
import {
	defaultDevice,
	getWebGPUDevice,
	init,
	jit,
	lax,
	nn,
	numpy as np,
	random,
} from "@jax-js/jax";
import type { Array as JaxArray, Device } from "@jax-js/jax";

export const BATCH = 4;

function partUrls(stem: string, count: number): string[] {
	return Array.from(
		{ length: count },
		(_, index) => `/un0/${stem}.safetensors.part${index}`,
	);
}

// Cloudflare serves static assets up to 25 MiB per file, so larger
// checkpoints are stored as split parts and reassembled after download.
// CIFAR-10 checkpoints are fp32; ImageNet-64 checkpoints are fp16 on disk
// (upcast to f32 at load) to halve their size.
export const CHECKPOINTS = {
	"cifar10/n1024": {
		family: "cifar10",
		short: "n1024",
		n: 1024,
		params: "1.3M",
		fid: 11.16,
		size: "5 MB",
		imageSize: 32,
		numClasses: 10,
		urls: ["/un0/cifar10_n1024.safetensors"],
		bytes: 5159380,
	},
	"cifar10/n2048": {
		family: "cifar10",
		short: "n2048",
		n: 2048,
		params: "4.9M",
		fid: 9.38,
		size: "20 MB",
		imageSize: 32,
		numClasses: 10,
		urls: ["/un0/cifar10_n2048.safetensors"],
		bytes: 19770620,
	},
	"cifar10/n4096": {
		family: "cifar10",
		short: "n4096",
		n: 4096,
		params: "19.4M",
		fid: 8.86,
		size: "78 MB",
		imageSize: 32,
		numClasses: 10,
		urls: partUrls("cifar10_n4096", 4),
		bytes: 77738372,
	},
	"imagenet64/n6656": {
		family: "imagenet64",
		short: "n6656",
		n: 6656,
		params: "57M",
		fid: 8.36,
		size: "114 MB",
		imageSize: 64,
		numClasses: 1000,
		urls: partUrls("imagenet64_n6656", 6),
		bytes: 114349586,
	},
	"imagenet64/n10240": {
		family: "imagenet64",
		short: "n10240",
		n: 10240,
		params: "130M",
		fid: 8.04,
		size: "260 MB",
		imageSize: 64,
		numClasses: 1000,
		urls: partUrls("imagenet64_n10240", 13),
		bytes: 259603122,
	},
	"imagenet64/n16384": {
		family: "imagenet64",
		short: "n16384",
		n: 16384,
		params: "322M",
		fid: 6.74,
		size: "645 MB",
		imageSize: 64,
		numClasses: 1000,
		urls: partUrls("imagenet64_n16384", 33),
		bytes: 644891170,
	},
} as const;
export type CheckpointName = keyof typeof CHECKPOINTS;
export const DEFAULT_CHECKPOINT: CheckpointName = "cifar10/n1024";

/** Download progress: bytes received so far out of the checkpoint's total. */
export type ProgressCallback = (loadedBytes: number, totalBytes: number) => void;

interface TensorSlice {
	data: Float32Array<ArrayBuffer>;
	shape: number[];
}

export interface ParsedSafetensors {
	tensors: Map<string, TensorSlice>;
	metadata: Record<string, string>;
}

/** Upcast IEEE 754 half-precision bits to a float32 array. */
function f16ToF32(half: Uint16Array): Float32Array<ArrayBuffer> {
	const out = new Float32Array(half.length);
	const bits = new Uint32Array(out.buffer);

	for (let i = 0; i < half.length; i++) {
		const h = half[i];
		const sign = (h & 0x8000) << 16;
		const exponent = (h >> 10) & 0x1f;
		const mantissa = h & 0x3ff;

		if (exponent === 0) {
			// Zero or subnormal: value = mantissa * 2^-24.
			out[i] = mantissa * 2 ** -24;
			if (sign) {
				out[i] = -out[i];
			}
		} else if (exponent === 31) {
			bits[i] = sign | 0x7f800000 | (mantissa << 13); // inf / nan
		} else {
			bits[i] = sign | ((exponent + 112) << 23) | (mantissa << 13);
		}
	}
	return out;
}

/** Parse the f32/f16 tensors of a safetensors buffer (f16 is upcast to f32). */
export function parseSafetensors(buffer: ArrayBuffer): ParsedSafetensors {
	const headerSize = Number(new DataView(buffer).getBigUint64(0, true));
	const header = JSON.parse(
		new TextDecoder().decode(new Uint8Array(buffer, 8, headerSize)),
	) as Record<
		string,
		{ dtype: string; shape: number[]; data_offsets: [number, number] }
	>;
	const dataStart = 8 + headerSize;
	const tensors = new Map<string, TensorSlice>();
	let metadata: Record<string, string> = {};

	for (const [name, info] of Object.entries(header)) {
		if (name === "__metadata__") {
			metadata = info as unknown as Record<string, string>;
			continue;
		}
		if (info.dtype !== "F32" && info.dtype !== "F16") {
			continue;
		}
		const begin = dataStart + info.data_offsets[0];
		const byteLength = info.data_offsets[1] - info.data_offsets[0];
		const elementBytes = info.dtype === "F16" ? 2 : 4;
		// View in place when aligned (the common case: safetensors pads the
		// header to 8 bytes); fall back to a copying slice otherwise. Views
		// avoid a second full-size transient copy of gigabyte checkpoints.
		const aligned = begin % elementBytes === 0;
		const raw = aligned ? buffer : buffer.slice(begin, begin + byteLength);
		const offset = aligned ? begin : 0;
		const data =
			info.dtype === "F16"
				? f16ToF32(new Uint16Array(raw, offset, byteLength / 2))
				: aligned
					? new Float32Array(buffer.slice(begin, begin + byteLength))
					: new Float32Array(raw, 0, byteLength / 4);

		tensors.set(name, { data, shape: info.shape });
	}
	return { tensors, metadata };
}

interface Model {
	n: number;
	nCond: number;
	steps: number;
	integrationTime: number;
	imageSize: number;
	numClasses: number;
	Kdrive: JaxArray;
	/** Every backend weight array, for disposal when the model is evicted. */
	weights: JaxArray[];
	eulerStep: (state: JaxArray, drive: JaxArray, dt: JaxArray) => JaxArray;
	readout: (phases: JaxArray) => JaxArray;
	decoder: (feat: JaxArray) => JaxArray;
}

const weightsBuffers = new Map<CheckpointName, ArrayBuffer>();
const modelPromises = new Map<CheckpointName, Promise<Model>>();
const builtModels = new Map<CheckpointName, Model>(); // insertion order = LRU
let backend: Device = "wasm";

// Keep total resident weight memory bounded: switching to a big model evicts
// older ones (least recently used first). Evicted models re-download from the
// browser's HTTP cache and recompile on next use. Resident bytes are fp32 on
// the backend: file size for fp32 checkpoints, double for fp16.
const RESIDENT_BUDGET_BYTES = 400_000_000;

function residentBytes(name: CheckpointName): number {
	const { family, bytes } = CHECKPOINTS[name];
	return family === "cifar10" ? bytes : 2 * bytes;
}

/** Whether a checkpoint is loaded and resident (switching to it is instant). */
export function isModelResident(name: CheckpointName): boolean {
	return builtModels.has(name);
}

function evictForBudget(incoming: CheckpointName) {
	let resident = 0;

	for (const name of builtModels.keys()) {
		resident += residentBytes(name);
	}
	for (const [name, model] of builtModels) {
		if (resident + residentBytes(incoming) <= RESIDENT_BUDGET_BYTES) {
			break;
		}
		if (name === incoming) {
			continue;
		}
		for (const array of model.weights) {
			while (array.refCount > 0) {
				array.dispose();
			}
		}
		builtModels.delete(name);
		modelPromises.delete(name);
		resident -= residentBytes(name);
	}
}

/** The backend the models run on ("webgpu" when available, else "wasm"). */
export function getBackend(): Device {
	return backend;
}

interface GPUDeviceLike {
	limits: { maxBufferSize: number; maxStorageBufferBindingSize: number };
	lost?: Promise<{ message?: string }>;
	addEventListener?: (
		type: string,
		listener: (event: { error?: { message?: string } }) => void,
	) => void;
}

let backendPromise: Promise<Device> | null = null;
let deviceLostHandler: ((message: string) => void) | null = null;

/**
 * Called when the WebGPU device is lost or reports an uncaptured error
 * (typically GPU memory exhaustion that no upfront check can rule out).
 */
export function setDeviceLostHandler(handler: (message: string) => void) {
	deviceLostHandler = handler;
}

function handleDeviceLost(message: string) {
	// Every GPU buffer died with the device; drop the caches so nothing tries
	// to reuse them. Recovery needs a reload, which the message says.
	for (const model of builtModels.values()) {
		for (const array of model.weights) {
			try {
				while (array.refCount > 0) {
					array.dispose();
				}
			} catch {
				// The backend may already consider the buffer gone.
			}
		}
	}
	builtModels.clear();
	modelPromises.clear();
	deviceLostHandler?.(message);
}

/** Initialize the compute backend once; safe to call repeatedly. */
export function ensureBackend(): Promise<Device> {
	if (!backendPromise) {
		backendPromise = buildBackend();
		// A rejected init must not be cached, or every retry fails instantly.
		backendPromise.catch(() => {
			backendPromise = null;
		});
	}
	return backendPromise;
}

function buildBackend(): Promise<Device> {
	return init("webgpu", "wasm").then((available) => {
		backend = available.includes("webgpu") ? "webgpu" : "wasm";
		defaultDevice(backend);
		if (backend === "webgpu") {
			const device = getWebGPUDevice() as unknown as GPUDeviceLike;

			device.lost?.then((info) =>
				handleDeviceLost(
					`The GPU device was lost (${info?.message || "no details"}). ` +
						`This usually means it ran out of memory — reload and try a smaller model.`,
				),
			);
			device.addEventListener?.("uncapturederror", (event) => {
				deviceLostHandler?.(
					`GPU error: ${event.error?.message ?? "unknown"}. ` +
						`If generation stops working, reload and try a smaller model.`,
				);
			});
		}
		return backend;
	});
}

export interface SupportEnv {
	backend: Device;
	maxBufferSize: number;
	maxStorageBufferBindingSize: number;
	deviceMemoryGb: number | null;
	saveData: boolean;
	effectiveType: string | null;
}

export interface ModelSupport {
	/** Hard gate: false means the model cannot run on this device. */
	supported: boolean;
	reason: string | null;
	/** Soft cautions worth showing near the controls. */
	warnings: string[];
	/** Things the user should explicitly accept before a run starts. */
	confirmations: string[];
}

/** Snapshot the environment used by checkSupport. Call after ensureBackend. */
export function currentSupportEnv(): SupportEnv {
	const limits =
		backend === "webgpu"
			? (getWebGPUDevice() as unknown as GPUDeviceLike).limits
			: { maxBufferSize: Infinity, maxStorageBufferBindingSize: Infinity };
	const nav =
		typeof navigator !== "undefined"
			? (navigator as {
					deviceMemory?: number;
					connection?: { saveData?: boolean; effectiveType?: string };
				})
			: undefined;

	return {
		backend,
		maxBufferSize: limits.maxBufferSize,
		maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
		deviceMemoryGb: nav?.deviceMemory ?? null,
		saveData: nav?.connection?.saveData ?? false,
		effectiveType: nav?.connection?.effectiveType ?? null,
	};
}

const gb = (bytes: number) => `${(bytes / 1e9).toFixed(1)} GB`;

// Thresholds for the soft checks, in resident (fp32) weight bytes.
const WASM_SLOW_BYTES = 200_000_000;
const WASM_CONFIRM_BYTES = 500_000_000;
const LOW_RAM_MODEL_BYTES = 400_000_000;
const LOW_RAM_GB = 4;
const NETWORK_CONFIRM_DOWNLOAD_BYTES = 100_000_000;

/**
 * Check whether a checkpoint can run here, before anything downloads.
 * The hard gate is the n×n coupling matrix, which must exist as (and bind
 * as) a single GPU buffer of 4n² bytes.
 */
export function checkSupport(
	name: CheckpointName,
	env: SupportEnv = currentSupportEnv(),
): ModelSupport {
	const info = CHECKPOINTS[name];
	const requiredBytes = 4 * info.n * info.n;
	const resident = residentBytes(name);
	const support: ModelSupport = {
		supported: true,
		reason: null,
		warnings: [],
		confirmations: [],
	};

	if (env.backend === "webgpu") {
		const allowed = Math.min(env.maxBufferSize, env.maxStorageBufferBindingSize);

		if (requiredBytes > allowed) {
			support.supported = false;
			support.reason =
				`${info.short} needs a ${gb(requiredBytes)} GPU buffer; ` +
				`this device allows ${gb(allowed)}`;
			return support;
		}
	} else {
		if (resident >= WASM_SLOW_BYTES) {
			support.warnings.push(
				`No WebGPU on this device: ${info.short} runs on the CPU and will be slow.`,
			);
		}
		if (resident >= WASM_CONFIRM_BYTES) {
			support.confirmations.push(
				`Without WebGPU, ${info.short} can take minutes per batch on the CPU.`,
			);
		}
	}
	if (
		env.deviceMemoryGb !== null &&
		env.deviceMemoryGb <= LOW_RAM_GB &&
		resident >= LOW_RAM_MODEL_BYTES
	) {
		support.warnings.push(
			`This device reports ~${env.deviceMemoryGb} GB RAM; loading ${info.short} may fail.`,
		);
	}
	if (
		(env.saveData ||
			(env.effectiveType !== null && /(^|-)2g$|^3g$/.test(env.effectiveType))) &&
		info.bytes > NETWORK_CONFIRM_DOWNLOAD_BYTES
	) {
		support.confirmations.push(
			env.saveData
				? `Data saver is on and ${info.short} downloads ${info.size}.`
				: `This connection is slow and ${info.short} downloads ${info.size}.`,
		);
	}
	return support;
}

/** Supply the safetensors bytes directly (used by Node tests instead of fetch). */
export function provideWeights(name: CheckpointName, buffer: ArrayBuffer) {
	weightsBuffers.set(name, buffer);
}

// Part files are produced by `split -b 20000000`, so part i starts at a
// fixed offset and every part can stream straight into one shared buffer.
const PART_SPLIT_BYTES = 20_000_000;

async function fetchPartInto(
	url: string,
	target: Uint8Array<ArrayBuffer>,
	offset: number,
	onBytes: (delta: number) => void,
	signal: AbortSignal,
): Promise<void> {
	const response = await fetch(url, { signal });

	if (!response.ok) {
		throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`);
	}
	if (!response.body) {
		const bytes = new Uint8Array(await response.arrayBuffer());

		target.set(bytes, offset); // RangeError if larger than expected
		onBytes(bytes.byteLength);
		return;
	}
	const reader = response.body.getReader();
	let position = offset;

	for (;;) {
		const { done, value } = await reader.read();

		if (done) {
			break;
		}
		target.set(value, position); // RangeError if larger than expected
		position += value.byteLength;
		onBytes(value.byteLength);
	}
}

async function fetchWeights(
	name: CheckpointName,
	onProgress?: ProgressCallback,
): Promise<ArrayBuffer> {
	const provided = weightsBuffers.get(name);

	if (provided) {
		return provided;
	}
	const { urls, bytes } = CHECKPOINTS[name];
	const target = new Uint8Array(new ArrayBuffer(bytes));
	const controller = new AbortController();
	let loaded = 0;
	let failed = false;
	const report = (delta: number) => {
		loaded += delta;
		if (!failed) {
			onProgress?.(loaded, bytes);
		}
	};

	activeDownload = controller;
	try {
		await Promise.all(
			urls.map((url, index) =>
				fetchPartInto(
					url,
					target,
					index * PART_SPLIT_BYTES,
					report,
					controller.signal,
				),
			),
		);
	} catch (error) {
		// Stop sibling part streams so a failed attempt doesn't keep spending
		// bandwidth or firing progress reports into a retry's UI.
		failed = true;
		controller.abort();
		throw error;
	} finally {
		if (activeDownload === controller) {
			activeDownload = null;
		}
	}
	if (loaded !== bytes) {
		throw new Error(`Downloaded ${loaded} bytes for ${name}, expected ${bytes}`);
	}
	return target.buffer;
}

let activeDownload: AbortController | null = null;

/** Abort the in-flight weights download, if any. The pending generate rejects. */
export function cancelDownload() {
	activeDownload?.abort();
}

function ensureModel(
	name: CheckpointName,
	onProgress?: ProgressCallback,
): Promise<Model> {
	let promise = modelPromises.get(name);

	if (!promise) {
		// Free older models first so the incoming build has room. Safe because
		// callers run generates serially; nothing is mid-flight on the evictees.
		evictForBudget(name);
		promise = buildModel(name, onProgress).then((model) => {
			// Only cache if this build is still current — a device loss clears
			// modelPromises mid-flight, and a model finished on a dead device
			// must not re-enter the cache as a zombie.
			if (modelPromises.get(name) === promise) {
				builtModels.set(name, model);
			} else {
				for (const array of model.weights) {
					try {
						while (array.refCount > 0) {
							array.dispose();
						}
					} catch {
						// The backend may already consider the buffer gone.
					}
				}
			}
			return model;
		});
		// A rejected build must not be cached, or every retry fails instantly.
		promise.catch(() => {
			modelPromises.delete(name);
		});
		modelPromises.set(name, promise);
	} else {
		const model = builtModels.get(name);

		if (model) {
			// Refresh LRU recency (Map preserves insertion order).
			builtModels.delete(name);
			builtModels.set(name, model);
		}
	}
	return promise;
}

async function buildModel(
	name: CheckpointName,
	onProgress?: ProgressCallback,
): Promise<Model> {
	await ensureBackend();

	// Hard-gate on GPU limits BEFORE downloading anything: the required buffer
	// size is known statically from the registry's oscillator count.
	const support = checkSupport(name);

	if (!support.supported) {
		throw new Error(`${support.reason}. Try a smaller model.`);
	}
	const buffer = await fetchWeights(name, onProgress);
	const { tensors, metadata } = parseSafetensors(buffer);

	// The raw checkpoint bytes are fully copied out by the parse; drop any
	// provided buffer so it isn't pinned for the rest of the session.
	weightsBuffers.delete(name);

	// The port implements exactly the configs the released checkpoints use;
	// fail loudly if a future export deviates.
	const relativization = metadata.relativization;

	if (
		metadata.solver !== "euler" ||
		(relativization !== "mean_relative" && relativization !== "ref_oscillator")
	) {
		throw new Error(
			`Unsupported checkpoint config: ${metadata.solver}/${relativization}`,
		);
	}
	const { imageSize, numClasses } = CHECKPOINTS[name];
	const n = Number(metadata.n);
	const nCond = Number(metadata.n_cond);
	const steps = Number(metadata.num_steps);
	const integrationTime = Number(metadata.integration_time);
	const numBlocks = Number(metadata.num_blocks);
	const decChannels = Number(metadata.decoder_in_channels);

	if (
		!(n > 0 && nCond > 0 && steps > 0 && integrationTime > 0) ||
		n !== CHECKPOINTS[name].n ||
		Number(metadata.num_classes) !== numClasses ||
		2 * n !== decChannels * 16 ||
		4 * 2 ** numBlocks !== imageSize
	) {
		throw new Error(`Unsupported checkpoint dimensions in ${name}`);
	}

	// Every created weight is tracked so a failure partway through loading
	// (missing tensor, shape mismatch) doesn't strand the earlier arrays.
	const createdWeights: JaxArray[] = [];

	function tensor(tensorName: string, expectedShape: number[]): JaxArray {
		const slice = tensors.get(tensorName);

		if (!slice) {
			throw new Error(`Missing tensor ${tensorName} in checkpoint`);
		}
		if (
			slice.shape.length !== expectedShape.length ||
			slice.shape.some((dim, i) => dim !== expectedShape[i])
		) {
			throw new Error(
				`Tensor ${tensorName} has shape [${slice.shape}], expected [${expectedShape}]`,
			);
		}
		const array = np.array(slice.data, { shape: slice.shape });

		createdWeights.push(array);
		return array;
	}

	// Conv biases are stored as (C); reshape for NCHW broadcast at load time.
	function bias(tensorName: string, channels: number): JaxArray {
		const array = tensor(tensorName, [channels]).reshape([1, channels, 1, 1]);

		createdWeights.push(array);
		return array;
	}

	// Cascade halves channels per block with a floor of 32, then maps to RGB.
	const blockWeights: {
		wa: JaxArray;
		ba: JaxArray;
		wb: JaxArray;
		bb: JaxArray;
	}[] = [];
	let K: JaxArray;
	let omega: JaxArray;
	let Kcond: JaxArray;
	let omegaCond: JaxArray;
	let Kdrive: JaxArray;
	let outWeight: JaxArray;
	let outBias: JaxArray;

	try {
		K = tensor("K", [n, n]); // diagonal pre-zeroed at export
		omega = tensor("omega", [1, n]);
		Kcond = tensor("K_cond", [nCond, nCond]); // diagonal pre-zeroed
		omegaCond = tensor("omega_cond", [1, nCond]);
		Kdrive = tensor("K_drive", [numClasses, n, nCond]);

		let channels = decChannels;

		for (let i = 0; i < numBlocks; i++) {
			const next = Math.max(channels >> 1, 32);

			blockWeights.push({
				wa: tensor(`dec.${i}.a.weight`, [next, channels, 3, 3]),
				ba: bias(`dec.${i}.a.bias`, next),
				wb: tensor(`dec.${i}.b.weight`, [next, next, 3, 3]),
				bb: bias(`dec.${i}.b.bias`, next),
			});
			channels = next;
		}
		outWeight = tensor("out.weight", [3, channels, 3, 3]);
		outBias = bias("out.bias", 3);
	} catch (error) {
		for (const array of createdWeights) {
			if (array.refCount > 0) {
				array.dispose();
			}
		}
		throw error;
	}
	// All weights now live on the backend; release the JS-side fp32 copies
	// (the map is captured by this scope's closures, so it must be emptied).
	tensors.clear();

	// dθ/dt = ω + cosθ·(sinθ@Kᵀ) − sinθ·(cosθ@Kᵀ)
	function kuramotoVelocity(theta: JaxArray, omega_: JaxArray, K_: JaxArray) {
		const s = np.sin(theta.ref);
		const c = np.cos(theta);
		const ws = np.matmul(s.ref, K_.ref.transpose());
		const wc = np.matmul(c.ref, K_.transpose());
		return omega_.add(c.mul(ws)).sub(s.mul(wc));
	}

	function velocity(state: JaxArray, drive: JaxArray) {
		const thM = state.ref.slice([], [0, n]);
		const thC = state.slice([], [n, n + nCond]);
		let velM = kuramotoVelocity(thM.ref, omega.ref, K.ref);
		const velC = kuramotoVelocity(thC.ref, omegaCond.ref, Kcond.ref);
		// one-way class drive: einsum("bnm,bm->bn", drive, sin/cos(theta_cond))
		const sC = np.sin(thC.ref);
		const cC = np.cos(thC);
		const dSin = np.einsum("bnm,bm->bn", drive.ref, sC);
		const dCos = np.einsum("bnm,bm->bn", drive, cC);
		velM = velM.add(np.cos(thM.ref).mul(dSin)).sub(np.sin(thM).mul(dCos));
		return np.concatenate([velM, velC], 1);
	}

	// These checkpoints integrate with fixed-step Euler (torchdiffeq "euler").
	const eulerStep = jit(function eulerStep(
		state: JaxArray,
		drive: JaxArray,
		dt: JaxArray,
	) {
		const v = velocity(state.ref, drive);
		return state.add(v.mul(dt));
	});

	// Relativization (per checkpoint config) + sin_cos encoding. mean_relative
	// subtracts the mean phase; ref_oscillator subtracts oscillator 0's phase.
	function readout(phases: JaxArray) {
		const rel =
			relativization === "mean_relative"
				? phases.ref.sub(np.mean(phases, 1, { keepdims: true }))
				: phases.ref.sub(phases.slice([], [0, 1]));
		return np.concatenate([np.sin(rel.ref), np.cos(rel)], 1);
	}

	function upsample2x(x: JaxArray) {
		return np.repeat(np.repeat(x, 2, 2), 2, 3);
	}
	function conv3x3(x: JaxArray, w: JaxArray, b: JaxArray) {
		return lax
			.convGeneralDilated(x, w.ref, [1, 1], [
				[1, 1],
				[1, 1],
			])
			.add(b.ref);
	}
	const decoder = jit(function decoder(feat: JaxArray) {
		let x = feat.reshape([BATCH, decChannels, 4, 4]);

		for (const { wa, ba, wb, bb } of blockWeights) {
			x = nn.leakyRelu(conv3x3(upsample2x(x), wa, ba), 0.2);
			x = nn.leakyRelu(conv3x3(x, wb, bb), 0.2);
		}
		x = np.tanh(conv3x3(x, outWeight, outBias));
		return x.reshape([BATCH, 3 * imageSize * imageSize]);
	});

	return {
		n,
		nCond,
		steps,
		integrationTime,
		imageSize,
		numClasses,
		Kdrive,
		weights: createdWeights,
		eulerStep,
		readout,
		decoder,
	};
}

export interface GenerateResult {
	/** Flat (BATCH, 3·imageSize·imageSize) NCHW pixels in [-1, 1]. */
	pixels: Float32Array;
	/** Flat (BATCH, n+n_cond) phases after integration. */
	finalState: Float32Array;
	/** Side length of each generated image (32 or 64). */
	imageSize: number;
	integrateMs: number;
	decodeMs: number;
}

async function runForward(
	model: Model,
	state0: JaxArray,
	classIds: number[],
): Promise<GenerateResult> {
	const { n, steps, integrationTime, Kdrive, eulerStep, readout, decoder } =
		model;

	const classIdsArray = np.array(classIds, { dtype: np.int32 });
	const drive = np.take(Kdrive.ref, classIdsArray, 0); // (BATCH, n, n_cond)
	const dt = np.array(integrationTime / steps);
	let state = state0;
	let phases: JaxArray | null = null;
	let feat: JaxArray | null = null;
	let img: JaxArray | null = null;

	try {
		const t0 = performance.now();
		for (let i = 0; i < steps; i++) {
			state = eulerStep(state, drive.ref, dt.ref);
		}
		const finalState = (await state.ref.data()) as Float32Array;
		phases = state.slice([], [0, n]);
		feat = readout(phases);
		const t1 = performance.now();
		img = decoder(feat);
		const pixels = (await img.data()) as Float32Array; // data() consumes img
		const t2 = performance.now();

		return {
			pixels,
			finalState,
			imageSize: model.imageSize,
			integrateMs: t1 - t0,
			decodeMs: t2 - t1,
		};
	} finally {
		// On success every array here is already consumed (refCount 0); on an
		// error path this reclaims whatever the throw left behind. The while
		// loop also covers .ref handles created before a throwing call.
		for (const array of [drive, dt, state, phases, feat, img]) {
			while (array && array.refCount > 0) {
				array.dispose();
			}
		}
	}
}

export async function generate(
	checkpoint: CheckpointName,
	classId: number,
	seed: number,
	onProgress?: ProgressCallback,
): Promise<GenerateResult> {
	const model = await ensureModel(checkpoint, onProgress);
	const state = random.uniform(random.key(seed), [BATCH, model.n + model.nCond], {
		minval: -Math.PI,
		maxval: Math.PI,
	});

	return runForward(model, state, new Array(BATCH).fill(classId));
}

/** Run the forward pass from a fixed initial state (for parity testing). */
export async function generateFromInitialState(
	checkpoint: CheckpointName,
	initialState: Float32Array<ArrayBuffer>,
	classIds: number[],
): Promise<GenerateResult> {
	const model = await ensureModel(checkpoint);

	if (
		initialState.length !== BATCH * (model.n + model.nCond) ||
		classIds.length !== BATCH
	) {
		throw new Error("initialState must be (BATCH, n+n_cond), classIds (BATCH,)");
	}
	const state = np.array(initialState, {
		shape: [BATCH, model.n + model.nCond],
	});

	return runForward(model, state, classIds);
}
