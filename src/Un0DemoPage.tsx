import { useCallback, useEffect, useRef, useState } from "react";
import styles from "./Un0DemoPage.module.css";
import { IMAGENET_CLASSES } from "./imagenetClasses";
import {
	BATCH,
	CHECKPOINTS,
	DEFAULT_CHECKPOINT,
	cancelDownload,
	checkSupport,
	ensureBackend,
	generate,
	getBackend,
	isModelResident,
	setDeviceLostHandler,
} from "./un0";
import type { CheckpointName, ModelSupport } from "./un0";

type Family = "cifar10" | "imagenet64";

const checkpointEntries = Object.entries(CHECKPOINTS) as [
	CheckpointName,
	(typeof CHECKPOINTS)[CheckpointName],
][];

const FAMILIES: { family: Family; label: string }[] = [
	{ family: "cifar10", label: "CIFAR-10" },
	{ family: "imagenet64", label: "ImageNet-64" },
];

function familyModels(family: Family): CheckpointName[] {
	return checkpointEntries
		.filter(([, info]) => info.family === family)
		.map(([name]) => name);
}

const CIFAR10_CLASSES = [
	"airplane",
	"automobile",
	"bird",
	"cat",
	"deer",
	"dog",
	"frog",
	"horse",
	"ship",
	"truck",
];

function classNamesFor(family: Family): string[] {
	return family === "cifar10" ? CIFAR10_CLASSES : IMAGENET_CLASSES;
}

function drawBatch(
	pixels: Float32Array,
	imageSize: number,
	canvases: (HTMLCanvasElement | null)[],
) {
	const planeSize = imageSize * imageSize;
	for (let b = 0; b < BATCH; b++) {
		const canvas = canvases[b];
		const context = canvas?.getContext("2d");

		if (!canvas || !context) {
			continue;
		}

		// Sized imperatively (not via JSX attrs): React re-assigning width or
		// height would clear the bitmap on every render.
		if (canvas.width !== imageSize || canvas.height !== imageSize) {
			canvas.width = imageSize;
			canvas.height = imageSize;
		}
		const imageData = context.createImageData(imageSize, imageSize);
		const base = b * 3 * planeSize;

		for (let i = 0; i < planeSize; i++) {
			for (let channel = 0; channel < 3; channel++) {
				const value = pixels[base + channel * planeSize + i];

				// [-1, 1] tanh output -> [0, 255]
				imageData.data[i * 4 + channel] = Math.round(
					(Math.min(1, Math.max(-1, value)) + 1) * 127.5,
				);
			}
			imageData.data[i * 4 + 3] = 255;
		}
		context.putImageData(imageData, 0, 0);
	}
}

function Un0DemoPage() {
	const [mode, setMode] = useState<"single" | "compare">("single");
	const [checkpoint, setCheckpoint] =
		useState<CheckpointName>(DEFAULT_CHECKPOINT);
	const [classId, setClassId] = useState(0);
	const [busy, setBusy] = useState(true);
	const [loadingModel, setLoadingModel] = useState(true);
	const [progress, setProgress] = useState<number | null>(null);
	// The model currently downloading/generating (drives the status line).
	const [activeModel, setActiveModel] = useState<CheckpointName | null>(null);
	const [error, setError] = useState<string | null>(null);
	// The class whose samples are actually drawn on the canvases (labels must
	// describe what is shown, not the pending selection).
	const [drawnClass, setDrawnClass] = useState<string | null>(null);
	const [timings, setTimings] = useState<{
		integrateMs: number;
		decodeMs: number;
	} | null>(null);
	// Per-model generation time for compare rows; presence marks a drawn row.
	const [rowTimes, setRowTimes] = useState<
		Partial<Record<CheckpointName, number>>
	>({});
	const canvasesRef = useRef<(HTMLCanvasElement | null)[]>([]);
	const compareCanvasesRef = useRef<
		Partial<Record<CheckpointName, (HTMLCanvasElement | null)[]>>
	>({});
	const seedRef = useRef(0);
	const runningRef = useRef(false);
	// Per-checkpoint capability results, computed once after backend init.
	const [support, setSupport] = useState<Partial<
		Record<CheckpointName, ModelSupport>
	> | null>(null);
	// Restore keyboard focus after a run: disabling the focused control drops
	// focus to <body>, stranding keyboard users.
	const focusRef = useRef<HTMLElement | null>(null);
	const classDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
	// Confirmations the user already accepted this session, per checkpoint.
	const confirmedRef = useRef<Set<CheckpointName>>(new Set());

	useEffect(() => {
		if (!busy && focusRef.current?.isConnected) {
			focusRef.current.focus();
			focusRef.current = null;
		}
	}, [busy]);

	useEffect(
		() => () => {
			if (classDebounceRef.current !== null) {
				clearTimeout(classDebounceRef.current);
			}
		},
		[],
	);

	// Generate with one model. Draws into the single-mode canvases unless a
	// target row is given. Returns false when the sweep should stop (error or
	// user cancel).
	const generateOne = useCallback(
		async (
			model: CheckpointName,
			nextClassId: number,
			seed: number,
			rowCanvases?: (HTMLCanvasElement | null)[],
		): Promise<boolean> => {
			const needsLoad = !isModelResident(model);

			setActiveModel(model);
			setLoadingModel(needsLoad);
			// Start at 0% for a fresh download so the phase reads "Downloading"
			// from the first moment, not "Compiling" during connection setup.
			setProgress(needsLoad ? 0 : null);

			try {
				const result = await generate(model, nextClassId, seed, (loaded, total) => {
					// Integer percent so identical values skip re-renders.
					setProgress(Math.min(100, Math.floor((loaded / total) * 100)));
				});

				drawBatch(
					result.pixels,
					result.imageSize,
					rowCanvases ?? canvasesRef.current,
				);
				setDrawnClass(
					classNamesFor(CHECKPOINTS[model].family)[nextClassId],
				);
				setTimings({
					integrateMs: result.integrateMs,
					decodeMs: result.decodeMs,
				});
				if (rowCanvases) {
					setRowTimes((times) => ({
						...times,
						[model]: result.integrateMs + result.decodeMs,
					}));
				}
				return true;
			} catch (runError) {
				const message =
					runError instanceof Error ? runError.message : String(runError);

				// A user-cancelled download is not an error state.
				if (!/abort/i.test(message)) {
					setError(message);
				}
				return false;
			} finally {
				setLoadingModel(false);
				setProgress(null);
			}
		},
		[],
	);

	const run = useCallback(
		async (
			nextMode: "single" | "compare",
			nextCheckpoint: CheckpointName,
			nextClassId: number,
			seed: number,
		) => {
			// A pending debounced class run is superseded: every caller passes
			// the latest classId, and letting the timer fire later would rerun
			// with the stale mode/checkpoint captured in its closure.
			if (classDebounceRef.current !== null) {
				clearTimeout(classDebounceRef.current);
				classDebounceRef.current = null;
			}
			if (runningRef.current) {
				return;
			}
			// Plan which models will actually run: skip hard-unsupported ones,
			// and collect outstanding confirmations before anything starts.
			const supportMap = supportRef.current;
			const wanted =
				nextMode === "single"
					? [nextCheckpoint]
					: familyModels(CHECKPOINTS[nextCheckpoint].family);
			const runnable = wanted.filter(
				(model) => supportMap?.[model]?.supported !== false,
			);

			if (runnable.length === 0) {
				setError(supportMap?.[nextCheckpoint]?.reason ?? "Model unavailable");
				setBusy(false); // the initial mount run may still hold busy=true
				return;
			}
			const pending = runnable.filter(
				(model) =>
					!confirmedRef.current.has(model) &&
					(supportMap?.[model]?.confirmations.length ?? 0) > 0,
			);

			if (pending.length > 0) {
				const message = pending
					.flatMap((model) => supportMap?.[model]?.confirmations ?? [])
					.join("\n");

				if (!window.confirm(`${message}\n\nContinue?`)) {
					setBusy(false);
					return;
				}
				for (const model of pending) {
					confirmedRef.current.add(model);
				}
			}
			runningRef.current = true;
			focusRef.current =
				document.activeElement instanceof HTMLElement
					? document.activeElement
					: null;
			setBusy(true);
			setError(null);

			try {
				// A mode switch remounts the canvases; wait for React to commit the
				// new DOM before drawing, or a fast (resident-model) generation can
				// finish while the target canvases don't exist yet. The timeout
				// backstops hidden tabs, where rAF never fires.
				await new Promise<void>((resolve) => {
					const timer = setTimeout(resolve, 150);

					requestAnimationFrame(() =>
						requestAnimationFrame(() => {
							clearTimeout(timer);
							resolve();
						}),
					);
				});
				if (nextMode === "single") {
					await generateOne(nextCheckpoint, nextClassId, seed);
				} else {
					// Blank every row first so pending rows can't show a previous
					// run's samples under "not generated yet" labels.
					setRowTimes({});
					for (const model of runnable) {
						for (const canvas of compareCanvasesRef.current[model] ?? []) {
							canvas
								?.getContext("2d")
								?.clearRect(0, 0, canvas.width, canvas.height);
						}
					}
					for (const model of runnable) {
						const ok = await generateOne(
							model,
							nextClassId,
							seed,
							compareCanvasesRef.current[model],
						);

						if (!ok) {
							break;
						}
					}
				}
			} finally {
				runningRef.current = false;
				setBusy(false);
				setActiveModel(null);
			}
		},
		[generateOne],
	);

	// The support map must be readable inside run() without re-creating the
	// callback (state would be stale in the mount effect's closure).
	const supportRef = useRef<Partial<Record<CheckpointName, ModelSupport>> | null>(
		null,
	);

	useEffect(() => {
		let disposed = false;

		setDeviceLostHandler((message) => {
			if (!disposed) {
				setError(message);
			}
		});
		void (async () => {
			try {
				await ensureBackend();
			} catch (initError) {
				// Without a backend the page can do nothing; surface it instead
				// of sitting permanently busy with a blank status.
				if (!disposed) {
					setError(
						initError instanceof Error ? initError.message : String(initError),
					);
					setBusy(false);
					setLoadingModel(false);
				}
				return;
			}
			const map = Object.fromEntries(
				checkpointEntries.map(([name]) => [name, checkSupport(name)]),
			) as Record<CheckpointName, ModelSupport>;

			supportRef.current = map;
			if (!disposed) {
				setSupport(map);
			}
			await run("single", DEFAULT_CHECKPOINT, 0, 0);
		})();
		return () => {
			disposed = true;
		};
	}, [run]);

	const family = CHECKPOINTS[checkpoint].family as Family;
	const classNames = classNamesFor(family);
	const active = activeModel ? CHECKPOINTS[activeModel] : null;
	// Capability notices: hard-blocked models (always shown) plus soft
	// warnings for whichever models the current mode would run.
	const unavailable = checkpointEntries
		.filter(([name]) => support?.[name]?.supported === false)
		.map(([name]) => support?.[name]?.reason ?? name);
	const relevantWarnings = [
		...new Set(
			(mode === "single" ? [checkpoint] : familyModels(family)).flatMap(
				(model) => support?.[model]?.warnings ?? [],
			),
		),
	];
	// The percent lives next to the progress bar, NOT in this aria-live text,
	// so screen readers announce phase changes rather than every percent step.
	const status = error
		? `Error: ${error}`
		: busy && active
			? loadingModel
				? progress === null || progress >= 100
					? `Compiling ${active.short}…`
					: `Downloading ${active.short} (${active.size})…`
				: `Generating ${active.short}…`
			: busy || timings === null
				? ""
				: `Integrate ${timings.integrateMs.toFixed(1)} ms · decode ${timings.decodeMs.toFixed(1)} ms (${getBackend()})`;

	return (
		<main className={styles.page}>
			<section
				className={styles.demo}
				aria-label="Un-0 Kuramoto image generator"
			>
				<h1 className={styles.title}>Un-0</h1>

				<div className={styles.controls}>
					{mode === "single" ? (
						<>
							<label className={styles.classLabel} htmlFor="un0-model">
								Model
							</label>
							<select
								className={`${styles.classSelect} ${styles.primarySelect}`}
								id="un0-model"
								value={checkpoint}
								onChange={(event) => {
									const next = event.target.value as CheckpointName;
									// Class ids mean different things across families.
									const nextClassId =
										CHECKPOINTS[next].family === family ? classId : 0;

									setCheckpoint(next);
									setClassId(nextClassId);
									void run("single", next, nextClassId, seedRef.current);
								}}
								disabled={busy}
							>
								{FAMILIES.map(({ family: optFamily, label }) => (
									<optgroup key={optFamily} label={label}>
										{checkpointEntries
											.filter(([, info]) => info.family === optFamily)
											.map(([name, info]) => (
												<option
													key={name}
													value={name}
													disabled={support?.[name]?.supported === false}
												>
													{info.short} · {info.params} · {info.size} · FID {info.fid}
													{support?.[name]?.supported === false
														? " — unavailable"
														: ""}
												</option>
											))}
									</optgroup>
								))}
							</select>
						</>
					) : (
						<>
							<label className={styles.classLabel} htmlFor="un0-family">
								Family
							</label>
							<select
								className={`${styles.classSelect} ${styles.primarySelect}`}
								id="un0-family"
								value={family}
								onChange={(event) => {
									const nextFamily = event.target.value as Family;

									if (nextFamily === family) {
										return;
									}
									// Land on the family's first SUPPORTED model so single
									// mode never ends up parked on a disabled option.
									const candidates = familyModels(nextFamily);
									const nextCheckpoint =
										candidates.find(
											(model) =>
												supportRef.current?.[model]?.supported !== false,
										) ?? candidates[0];

									setCheckpoint(nextCheckpoint);
									setClassId(0);
									// The rows remount blank for the new family's models.
									setDrawnClass(null);
									setRowTimes({});
									void run("compare", nextCheckpoint, 0, seedRef.current);
								}}
								disabled={busy}
							>
								{FAMILIES.map(({ family: optFamily, label }) => (
									<option key={optFamily} value={optFamily}>
										{label}
									</option>
								))}
							</select>
						</>
					)}
					<label className={styles.classLabel} htmlFor="un0-class">
						Class
					</label>
					<select
						className={`${styles.classSelect} ${styles.classPicker}`}
						id="un0-class"
						value={classId}
						onChange={(event) => {
							const next = Number(event.target.value);

							setClassId(next);
							// Debounced: arrow-key browsing on a closed select fires a
							// change per keystroke; generating on each would lock the
							// control after one step (1000 ImageNet classes).
							if (classDebounceRef.current !== null) {
								clearTimeout(classDebounceRef.current);
							}
							classDebounceRef.current = setTimeout(() => {
								classDebounceRef.current = null;
								void run(mode, checkpoint, next, seedRef.current);
							}, 250);
						}}
						disabled={busy}
					>
						{classNames.map((name, id) => (
							<option key={id} value={id}>
								{id} — {name}
							</option>
						))}
					</select>
					<button
						className={styles.generateButton}
						type="button"
						onClick={() => {
							seedRef.current += 1;
							void run(mode, checkpoint, classId, seedRef.current);
						}}
						disabled={busy}
					>
						{mode === "compare" ? "Generate all" : "Generate"}
					</button>
					<label className={styles.toggleLabel}>
						<input
							type="checkbox"
							checked={mode === "compare"}
							onChange={(event) => {
								const nextMode = event.target.checked ? "compare" : "single";

								setMode(nextMode);
								// The canvases remount blank on a mode switch; clear the
								// drawn-state so labels/timings can't describe images that
								// are no longer visible if the rerun fails or is cancelled.
								setDrawnClass(null);
								setTimings(null);
								setRowTimes({});
								void run(nextMode, checkpoint, classId, seedRef.current);
							}}
							disabled={busy}
						/>
						Compare models
					</label>
				</div>

				{unavailable.length > 0 && (
					<p className={styles.compareNote}>
						Not available on this device: {unavailable.join("; ")}.
					</p>
				)}

				{relevantWarnings.length > 0 && (
					<p className={styles.compareNote}>{relevantWarnings.join(" ")}</p>
				)}

				{busy && loadingModel && progress !== null && (
					<div className={styles.progressRow}>
						<div
							className={styles.progressTrack}
							role="progressbar"
							aria-label={`Downloading ${active?.short ?? ""}`}
							aria-valuemin={0}
							aria-valuemax={100}
							aria-valuenow={progress}
						>
							<div
								className={styles.progressFill}
								style={{ width: `${progress}%` }}
							/>
						</div>
						<span className={styles.progressPct} aria-hidden="true">
							{progress}%
						</span>
						{progress < 100 && (
							<button
								className={styles.cancelButton}
								type="button"
								onClick={cancelDownload}
							>
								Cancel
							</button>
						)}
					</div>
				)}

				<p className={styles.status} aria-live="polite">
					{status}
				</p>

				{mode === "single" ? (
					<div className={styles.gallery}>
						{Array.from({ length: BATCH }, (_, index) => (
							<canvas
								key={index}
								className={styles.sample}
								ref={(element) => {
									canvasesRef.current[index] = element;
								}}
								role="img"
								aria-label={
									drawnClass
										? `Generated sample ${index + 1} for class ${drawnClass}`
										: `Sample ${index + 1}, not generated yet`
								}
							/>
						))}
					</div>
				) : (
					<div className={styles.compareGrid}>
						{familyModels(family).map((model) => {
							const info = CHECKPOINTS[model];
							const rowTime = rowTimes[model];

							if (support?.[model]?.supported === false) {
								return (
									<div key={model} className={styles.compareRow}>
										<div className={styles.rowLabel}>
											<strong>{info.short}</strong>
											{info.params} · FID {info.fid}
										</div>
										<p className={styles.rowUnavailable}>
											{support[model]?.reason}
										</p>
									</div>
								);
							}
							return (
								<div key={model} className={styles.compareRow}>
									<div className={styles.rowLabel}>
										<strong>{info.short}</strong>
										{info.params} · FID {info.fid}
										{rowTime !== undefined && (
											<span className={styles.rowTime}>
												{rowTime.toFixed(0)} ms
											</span>
										)}
									</div>
									{Array.from({ length: BATCH }, (_, index) => (
										<canvas
											key={index}
											className={styles.sample}
											ref={(element) => {
												(compareCanvasesRef.current[model] ??= [])[index] =
													element;
											}}
											role="img"
											aria-label={
												rowTime !== undefined && drawnClass
													? `${info.short} sample ${index + 1} for class ${drawnClass}`
													: `${info.short} sample ${index + 1}, not generated yet`
											}
										/>
									))}
								</div>
							);
						})}
					</div>
				)}

				<details className={styles.readMore}>
					<summary>
						Read More <span aria-hidden="true">⌄</span>
					</summary>
					<ul>
						<li>
							FID values: clean-FID for CIFAR-10, ADM FID for ImageNet-64
							(comparable within a family, not across).
						</li>
						<li>
							<a
								href="https://unconv.ai/blog/introducing-un-0-generating-images-with-coupled-oscillators/"
								target="_blank"
								rel="noopener noreferrer"
							>
								https://unconv.ai/blog/introducing-un-0-generating-images-with-coupled-oscillators/
							</a>
						</li>
						<li>
							<a
								href="https://github.com/unconv-ai/Un-0"
								target="_blank"
								rel="noopener noreferrer"
							>
								https://github.com/unconv-ai/Un-0
							</a>
						</li>
						<li>
							<a
								href="https://github.com/ekzhang/jax-js"
								target="_blank"
								rel="noopener noreferrer"
							>
								https://github.com/ekzhang/jax-js
							</a>
						</li>
					</ul>
				</details>
			</section>
		</main>
	);
}

export default Un0DemoPage;
