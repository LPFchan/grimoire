<script lang="ts">
	import Dashboard from './lib/Dashboard.svelte';
	import Login from './lib/Login.svelte';
	import { clearApiKey, loadConfig } from './lib/api';

	let signedIn = $state(loadConfig().apiKey !== '');
	let notice = $state<string | null>(null);

	function signOut(message: string | null = null) {
		clearApiKey();
		notice = message;
		signedIn = false;
	}
</script>

{#if signedIn}
	<Dashboard
		onUnauthorized={() => signOut('Your key was rejected — sign in again.')}
		onSignOut={() => signOut()}
	/>
{:else}
	<Login {notice} onAuthenticated={() => { notice = null; signedIn = true; }} />
{/if}
