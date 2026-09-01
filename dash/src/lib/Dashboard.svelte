<script lang="ts">
	import { apiFetch } from './api';

	type Scale = {
		min: number;
		/** null means "anchor the floor, let the top follow the data". */
		max: number | null;
	};

	type Series = {
		current: number | null;
		series: Array<number | null>;
		scale?: Scale | null;
	};

	type DashboardResponse = {
		window: string;
		from: number;
		to: number;
		bins: number;
		tokens: { input: Series; output: Series };
		cache: {
			read: {
				tokens: Series;
				cost: Series;
			};
			lifetime_tokens: number;
		};
		cost: {
			total: number;
			input: number;
			output: number;
			lifetime: number;
			series: Array<number>;
		};
		gpus: Array<{ index: number; temp: Series; power: Series; vram: Series; tokens_per_sec: Series }>;
		cpu: { temp: Series; power: Series };
		fans: { fan1: Series; fan2: Series };
		ram: { system: Series; container: Series };
		disk: Series;
		card_order: string[] | null;
	};

	const WINDOWS = [
		{ id: '5m', label: '5m', seconds: 300 },
		{ id: '15m', label: '15m', seconds: 900 },
		{ id: '1h', label: '1h', seconds: 3600 },
		{ id: '6h', label: '6h', seconds: 21600 },
		{ id: '24h', label: '24h', seconds: 86400 },
		{ id: '7d', label: '7d', seconds: 604800 },
		{ id: '30d', label: '30d', seconds: 2592000 },
		{ id: 'all', label: 'All', seconds: Number.POSITIVE_INFINITY }
	] as const;

	const BINS = 60;
	const MIN_POLL_MS = 1000;
	const MAX_POLL_MS = 30000;

	type CardKey = string;

	type CardDef = {
		key: CardKey;
		title: string;
		value: string;
		series: Array<number | null> | undefined;
		scale?: Scale | null;
		accent: string;
		formatFn: (n: number) => string;
	};

	let selectedWindow = $state<(typeof WINDOWS)[number]['id']>('1h');
	let data = $state<DashboardResponse | null>(null);
	let error = $state<string | null>(null);
	let loading = $state(false);
	let cardOrder = $state<CardKey[] | null>(null);
	let dragKey = $state<CardKey | null>(null);

	async function refresh() {
		loading = true;
		try {
			const result = await apiFetch<DashboardResponse>(
				`/stats/dashboard?window=${selectedWindow}`
			);
			data = result;
			if (result.card_order) {
				cardOrder = result.card_order;
			}
			error = null;
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
		}
	}

	// Refetch no faster than a quarter of one bin. On a 24h graph each point
	// covers 24 minutes, so polling every second only burns queries on the
	// gateway — which is the process also serving chat. Mirrors the payload
	// cache on the server side.
	let pollingMs = $derived.by(() => {
		const seconds = WINDOWS.find((w) => w.id === selectedWindow)?.seconds ?? 3600;
		if (!Number.isFinite(seconds)) return MAX_POLL_MS;
		return Math.min(Math.max((seconds / BINS / 4) * 1000, MIN_POLL_MS), MAX_POLL_MS);
	});

	$effect(() => {
		void selectedWindow;
		void refresh();
	});

	// A backgrounded tab polls nothing; it catches up when you come back.
	let visible = $state(true);

	$effect(() => {
		const sync = () => {
			visible = document.visibilityState === 'visible';
			if (visible) void refresh();
		};
		sync();
		document.addEventListener('visibilitychange', sync);
		return () => document.removeEventListener('visibilitychange', sync);
	});

	$effect(() => {
		if (!visible) return;
		const handle = setInterval(() => { void refresh(); }, pollingMs);
		return () => clearInterval(handle);
	});

	function fmtCost(n: number | null | undefined): string {
		const v = n ?? 0;
		if (v >= 100) return `$${v.toFixed(2)}`;
		if (v >= 1) return `$${v.toFixed(3)}`;
		if (v >= 0.01) return `$${v.toFixed(4)}`;
		if (v > 0) return `$${v.toFixed(6)}`;
		return '$0.00';
	}

	function fmtTokens(n: number | null | undefined): string {
		if (n == null) return '—';
		if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(2)}B`;
		if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
		if (n >= 1_000) return `${(n / 1_000).toFixed(2)}k`;
		return Number.isInteger(n) ? `${n}` : n.toFixed(0);
	}

	function fmtTemp(c: number | null | undefined): string {
		if (c == null) return '—';
		return `${Math.round(c)}°C`;
	}

	function fmtPower(w: number | null | undefined): string {
		if (w == null) return '—';
		return `${w.toFixed(1)} W`;
	}

	function fmtRpm(n: number | null | undefined): string {
		if (n == null) return '—';
		if (n >= 1000) return `${(n / 1000).toFixed(1)}k RPM`;
		return `${Math.round(n)} RPM`;
	}

	function fmtVram(mb: number | null | undefined): string {
		if (mb == null) return '—';
		if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
		return `${Math.round(mb)} MB`;
	}

	function fmtTps(n: number | null | undefined): string {
		if (n == null) return '—';
		return `${n.toFixed(1)} t/s`;
	}

	function fmtRam(mb: number | null | undefined): string {
		if (mb == null) return '—';
		if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
		return `${Math.round(mb)} MB`;
	}

	function fmtPct(n: number | null | undefined): string {
		if (n == null) return '—';
		return `${n.toFixed(0)}%`;
	}

	type Sparkline = { line: string; area: string; lo: number; hi: number };

	function buildSparkline(
		raw: Array<number | null> | undefined,
		width: number,
		height: number,
		scale?: Scale | null
	): Sparkline {
		const values = raw ?? [];
		if (values.length === 0) return { line: '', area: '', lo: 0, hi: 1 };
		let lo = Number.POSITIVE_INFINITY;
		let hi = Number.NEGATIVE_INFINITY;
		for (const v of values) {
			if (v == null || !Number.isFinite(v)) continue;
			if (v < lo) lo = v;
			if (v > hi) hi = v;
		}
		if (!Number.isFinite(lo)) {
			lo = 0;
			hi = 1;
		}

		// An absolute axis from the server wins. A null max still pins the floor,
		// which is what stops an all-zero window drawing as a filled block.
		if (scale) {
			lo = scale.min;
			hi = scale.max ?? Math.max(hi, scale.min);
		}

		if (lo === hi) {
			// A flat series still needs a range to divide by. With an absolute
			// axis the floor is meaningful, so grow upward and leave it alone —
			// expanding both ways would push an all-zero window back to the
			// middle of the card, which is what made idle token counts look
			// like activity.
			if (scale) {
				hi = lo + 1;
			} else {
				lo -= 1;
				hi += 1;
			}
		}
		const dx = values.length > 1 ? width / (values.length - 1) : 0;
		let lastY = height;
		let line = '';
		for (let i = 0; i < values.length; i++) {
			const v = values[i];
			const x = i * dx;
			const isNum = v != null && Number.isFinite(v);
			const y = isNum
				? Math.min(height, Math.max(0, height - (((v as number) - lo) / (hi - lo)) * height))
				: lastY;
			line += i === 0 ? `M${x.toFixed(2)},${y.toFixed(2)}` : ` L${x.toFixed(2)},${y.toFixed(2)}`;
			if (isNum) lastY = y;
		}
		const area = `${line} L${width.toFixed(2)},${height.toFixed(2)} L0,${height.toFixed(2)} Z`;
		return { line, area, lo, hi };
	}

	type HoverInfo = { idx: number; x: number; value: number };
	let hoverState = $state<Record<string, HoverInfo | undefined>>({});

	function handleSparkMouseMove(
		e: MouseEvent,
		cardKey: string,
		values: Array<number | null>,
		lo: number,
		hi: number
	) {
		const svg = e.currentTarget as SVGSVGElement;
		const rect = svg.getBoundingClientRect();
		const scaleX = SPARK_W / rect.width;
		const svgX = (e.clientX - rect.left) * scaleX;
		if (values.length === 0) return;
		const dx = values.length > 1 ? SPARK_W / (values.length - 1) : 0;
		let bestIdx = 0;
		let bestDist = Infinity;
		for (let i = 0; i < values.length; i++) {
			const dist = Math.abs(i * dx - svgX);
			if (dist < bestDist) { bestDist = dist; bestIdx = i; }
		}
		const val = values[bestIdx];
		if (val == null || !Number.isFinite(val)) {
			hoverState = { ...hoverState, [cardKey]: undefined };
			return;
		}
		hoverState = {
			...hoverState,
			[cardKey]: { idx: bestIdx, x: bestIdx * dx, value: val }
		};
	}

	function clearHover(cardKey: string) {
		hoverState = { ...hoverState, [cardKey]: undefined };
	}

	const SPARK_W = 280;
	const SPARK_H = 56;

	function buildCardDefs(d: DashboardResponse): CardDef[] {
		const cards: CardDef[] = [
			{ key: 'input_tokens', title: 'Input tokens', value: fmtTokens(d.tokens.input.current), series: d.tokens.input.series, scale: d.tokens.input.scale, accent: 'oklch(0.72 0.16 252)', formatFn: fmtTokens },
			{ key: 'output_tokens', title: 'Output tokens', value: fmtTokens(d.tokens.output.current), series: d.tokens.output.series, scale: d.tokens.output.scale, accent: 'oklch(0.74 0.16 152)', formatFn: fmtTokens },
		];

		if (d.cache) {
			cards.push({ key: 'cache_read_tokens', title: 'Cache read tokens', value: fmtTokens(d.cache.read.tokens.current), series: d.cache.read.tokens.series, scale: d.cache.read.tokens.scale, accent: 'oklch(0.76 0.18 42)', formatFn: fmtTokens });
			cards.push({ key: 'cache_read_cost', title: 'Cache read cost', value: fmtCost(d.cache.read.cost.current), series: d.cache.read.cost.series, scale: d.cache.read.cost.scale, accent: 'oklch(0.80 0.17 80)', formatFn: fmtCost });
		}

		cards.push(
			{ key: 'cpu_temp', title: 'CPU temp', value: fmtTemp(d.cpu.temp.current), series: d.cpu.temp.series, scale: d.cpu.temp.scale, accent: 'oklch(0.72 0.16 320)', formatFn: fmtTemp },
			{ key: 'cpu_power', title: 'CPU power', value: fmtPower(d.cpu.power.current), series: d.cpu.power.series, scale: d.cpu.power.scale, accent: 'oklch(0.75 0.18 120)', formatFn: fmtPower },
		);

		for (const gpu of d.gpus) {
			cards.push(
				{ key: `gpu${gpu.index}_temp`, title: `GPU${gpu.index} temp`, value: fmtTemp(gpu.temp.current), series: gpu.temp.series, scale: gpu.temp.scale, accent: 'oklch(0.72 0.18 30)', formatFn: fmtTemp },
				{ key: `gpu${gpu.index}_power`, title: `GPU${gpu.index} power`, value: fmtPower(gpu.power.current), series: gpu.power.series, scale: gpu.power.scale, accent: 'oklch(0.78 0.16 80)', formatFn: fmtPower },
				{ key: `gpu${gpu.index}_vram`, title: `GPU${gpu.index} VRAM`, value: fmtVram(gpu.vram.current), series: gpu.vram.series, scale: gpu.vram.scale, accent: 'oklch(0.65 0.18 270)', formatFn: fmtVram },
				{ key: `gpu${gpu.index}_tps`, title: `GPU${gpu.index} t/s`, value: fmtTps(gpu.tokens_per_sec.current), series: gpu.tokens_per_sec.series, scale: gpu.tokens_per_sec.scale, accent: 'oklch(0.70 0.16 60)', formatFn: fmtTps },
			);
		}

		cards.push(
			{ key: 'system_ram', title: 'System RAM', value: fmtRam(d.ram.system.current), series: d.ram.system.series, scale: d.ram.system.scale, accent: 'oklch(0.72 0.17 200)', formatFn: fmtRam },
			{ key: 'grimoire_ram', title: 'Grimoire RAM', value: fmtRam(d.ram.container.current), series: d.ram.container.series, scale: d.ram.container.scale, accent: 'oklch(0.74 0.16 222)', formatFn: fmtRam },
			{ key: 'disk_usage', title: 'Disk usage', value: fmtPct(d.disk.current), series: d.disk.series, scale: d.disk.scale, accent: 'oklch(0.70 0.15 160)', formatFn: fmtPct },
			{ key: 'fan1', title: 'Fan 1', value: fmtRpm(d.fans.fan1.current), series: d.fans.fan1.series, scale: d.fans.fan1.scale, accent: 'oklch(0.68 0.16 190)', formatFn: fmtRpm },
			{ key: 'fan2', title: 'Fan 2', value: fmtRpm(d.fans.fan2.current), series: d.fans.fan2.series, scale: d.fans.fan2.scale, accent: 'oklch(0.68 0.14 250)', formatFn: fmtRpm },
		);

		return cards;
	}

	let allCardDefs = $derived(data ? buildCardDefs(data) : []);
	let allCardKeys = $derived(allCardDefs.map(c => c.key));

	let visibleOrder = $derived.by(() => {
		const order = cardOrder;
		if (!order) return allCardKeys;
		const ordered = order.filter(k => allCardKeys.includes(k));
		const remaining = allCardKeys.filter(k => !order.includes(k));
		return [...ordered, ...remaining];
	});

	let cardDefMap = $derived(
		Object.fromEntries(allCardDefs.map(c => [c.key, c]))
	);

	let visibleCards = $derived(
		visibleOrder.map(k => cardDefMap[k]).filter(Boolean)
	);

	function handleDragStart(e: DragEvent, key: CardKey) {
		if (!e.dataTransfer) return;
		dragKey = key;
		e.dataTransfer.effectAllowed = 'move';
		e.dataTransfer.setData('text/plain', key);
	}

	function handleDragOver(e: DragEvent, key: CardKey) {
		e.preventDefault();
		if (!e.dataTransfer) return;
		e.dataTransfer.dropEffect = 'move';
	}

	function handleDrop(e: DragEvent, targetKey: CardKey) {
		e.preventDefault();
		if (!dragKey || dragKey === targetKey) {
			dragKey = null;
			return;
		}

		const newOrder = visibleOrder.filter(k => k !== dragKey);
		const targetIdx = newOrder.indexOf(targetKey);
		const dragIdx = visibleOrder.indexOf(dragKey);
		if (targetIdx === -1 || dragIdx === -1) {
			dragKey = null;
			return;
		}
		newOrder.splice(targetIdx, 0, dragKey);
		cardOrder = newOrder;
		dragKey = null;

		apiFetch('/stats/card-order', {
			method: 'PUT',
			body: JSON.stringify({ card_order: newOrder }),
		}).catch(() => {});
	}

	function handleDragEnd() {
		dragKey = null;
	}
</script>

<div class="flex h-full w-full flex-col overflow-auto px-6 py-6 md:px-10 md:py-8">
	<header class="sticky top-0 z-10 -mx-6 -mt-6 mb-8 flex flex-wrap items-center justify-end gap-4 px-6 py-3">
		<!-- Frosted rather than opaque: the header is sticky and cards scroll
		     underneath, so the pill needs to stay legible without painting a
		     full-width bar across the top of the page. -->
		<div
			class="inline-flex rounded-lg border bg-muted/60 p-1 shadow-md backdrop-blur-md"
			role="tablist"
			aria-label="Window"
		>
			{#each WINDOWS as w}
				<button
					type="button"
					role="tab"
					aria-selected={selectedWindow === w.id}
					class="rounded-md px-3 py-1 text-sm transition {selectedWindow === w.id
						? 'bg-background font-medium shadow-sm'
						: 'text-muted-foreground hover:text-foreground'}"
					onclick={() => (selectedWindow = w.id)}
				>
					{w.label}
				</button>
			{/each}
		</div>
	</header>

	{#if error}
		<div
			class="mb-4 rounded-md border border-destructive bg-destructive/10 px-4 py-3 text-sm text-destructive"
		>
			{error}
		</div>
	{/if}

	<section
		class="mb-8 flex flex-col items-center px-8 py-12 text-center"
	>
		<div class="text-xs font-medium uppercase tracking-widest text-muted-foreground">
			Lifetime cost
		</div>
		<div class="mt-3 text-6xl font-semibold tabular-nums md:text-7xl">
			{fmtCost(data?.cost.lifetime)}
		</div>
		{#if data}
			<div class="mt-3 text-sm text-muted-foreground">
				{selectedWindow === 'all'
					? `${fmtCost(data.cost.total)} across all recorded events`
					: `${fmtCost(data.cost.total)} in last ${WINDOWS.find((w) => w.id === selectedWindow)?.label ?? selectedWindow}`}
			</div>
		{:else if loading}
			<div class="mt-3 text-sm text-muted-foreground">Loading…</div>
		{/if}
	</section>

	<section
		class="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
		role="list"
	>
		{#snippet statCard(card: CardDef)}
			{@const spark = buildSparkline(card.series, SPARK_W, SPARK_H, card.scale)}
			{@const values = card.series ?? []}
			{@const hov = hoverState[card.key]}
			<div
				role="listitem"
				draggable="true"
				class="flex flex-col rounded-xl border bg-card/40 p-4 {dragKey === card.key ? 'opacity-50 ring-2 ring-primary' : ''}"
				ondragstart={(e) => handleDragStart(e, card.key)}
				ondragover={(e) => handleDragOver(e, card.key)}
				ondrop={(e) => handleDrop(e, card.key)}
				ondragend={handleDragEnd}
			>
				<div class="flex items-center justify-between">
					<div class="text-xs font-medium uppercase tracking-widest text-muted-foreground">
						{card.title}
					</div>
					<div class="cursor-grab text-xs text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" style="opacity: 0.3;">⠿</div>
				</div>
				<div class="my-2 text-3xl font-semibold tabular-nums transition-colors" class:text-primary={hov != null}>
					{hov != null ? card.formatFn(hov.value) : card.value}
				</div>
				<svg
					viewBox={`0 0 ${SPARK_W} ${SPARK_H}`}
					class="h-14 w-full cursor-crosshair"
					preserveAspectRatio="none"
					role="img"
					aria-label={`${card.title} over the selected window`}
					style:color={card.accent}
					onmousemove={(e) => handleSparkMouseMove(e, card.key, values, spark.lo, spark.hi)}
					onmouseleave={() => clearHover(card.key)}
				>
					{#if spark.area}
						<path d={spark.area} fill="currentColor" fill-opacity="0.18" />
						<path d={spark.line} fill="none" stroke="currentColor" stroke-width="1.5" />
					{/if}
					{#if hov != null}
						{@const dotY = Math.min(SPARK_H, Math.max(0, SPARK_H - ((hov.value - spark.lo) / (spark.hi - spark.lo)) * SPARK_H))}
						<line
							x1={hov.x} y1="0"
							x2={hov.x} y2={SPARK_H}
							stroke="currentColor"
							stroke-width="1"
							stroke-dasharray="3 3"
							opacity="0.5"
						/>
						<circle
							cx={hov.x} cy={dotY}
							r="4"
							fill="currentColor"
							stroke="var(--background)"
							stroke-width="1.5"
						/>
					{/if}
				</svg>
			</div>
		{/snippet}

		{#each visibleCards as card (card.key)}
			{@render statCard(card)}
		{/each}
	</section>
</div>
