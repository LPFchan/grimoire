<script lang="ts">
	import { apiFetch, loadConfig, saveConfig, UnauthorizedError } from './api';

	let { onAuthenticated, notice }: {
		onAuthenticated: () => void;
		notice?: string | null;
	} = $props();

	const initial = loadConfig();

	let apiKey = $state('');
	let apiBase = $state(initial.apiBase);
	let showAdvanced = $state(false);
	let error = $state<string | null>(null);
	let checking = $state(false);

	async function submit(e: SubmitEvent) {
		e.preventDefault();
		if (checking) return;

		checking = true;
		error = null;

		const previous = loadConfig();
		saveConfig({ apiBase: apiBase.trim(), apiKey: apiKey.trim() });

		try {
			// Cheapest window that still proves the key and CORS both work.
			await apiFetch('/stats/dashboard?window=5m');
			onAuthenticated();
		} catch (e) {
			saveConfig(previous);
			error =
				e instanceof UnauthorizedError
					? 'That key was rejected.'
					: e instanceof Error
						? e.message
						: String(e);
		} finally {
			checking = false;
		}
	}
</script>

<div class="grid h-full place-items-center px-4">
	<form class="w-full max-w-sm" onsubmit={submit}>
		<h1 class="text-2xl font-semibold">Grimoire Dashboard</h1>
		<p class="mt-2 text-sm text-muted-foreground">Sign in with your Grimoire API key.</p>

		{#if notice}
			<p class="mt-4 rounded-md border bg-muted/30 px-3 py-2 text-sm text-muted-foreground">
				{notice}
			</p>
		{/if}

		<input
			type="password"
			bind:value={apiKey}
			placeholder="API key"
			autocomplete="current-password"
			class="mt-6 w-full rounded-lg border bg-transparent px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-ring"
		/>

		{#if showAdvanced}
			<label class="mt-4 block text-xs font-medium tracking-widest text-muted-foreground uppercase">
				Gateway URL
				<input
					type="url"
					bind:value={apiBase}
					placeholder="https://chat.lost.plus"
					class="mt-2 w-full rounded-lg border bg-transparent px-3 py-2 text-sm normal-case tracking-normal outline-none focus:ring-2 focus:ring-ring"
				/>
			</label>
		{:else}
			<button
				type="button"
				class="mt-3 text-xs text-muted-foreground underline-offset-2 hover:underline"
				onclick={() => (showAdvanced = true)}
			>
				Using a different gateway?
			</button>
		{/if}

		{#if error}
			<p class="mt-4 rounded-md border border-destructive bg-destructive/10 px-3 py-2 text-sm text-destructive">
				{error}
			</p>
		{/if}

		<button
			type="submit"
			disabled={checking || apiKey.trim() === ''}
			class="mt-6 w-full rounded-lg bg-primary px-3 py-2 text-sm font-semibold text-primary-foreground transition disabled:opacity-50"
		>
			{checking ? 'Checking…' : 'Sign in'}
		</button>
	</form>
</div>
