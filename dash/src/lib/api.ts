/**
 * Talking to Grimoire from a different origin.
 *
 * The dashboard used to live inside the chat webui, so it was same-origin with
 * the gateway and the login cookie came along for free. Served from
 * dash.lost.plus it is cross-origin, so the cookie is not sent (it is
 * SameSite=Lax and scoped to the chat host). We hold the API key ourselves and
 * send it as a bearer token instead. Grimoire must allow this origin via
 * GRIMOIRE_CORS_ORIGINS.
 */

const STORAGE_KEY = 'grimoire-dash.config';

const DEFAULT_API_BASE = (
	import.meta.env.VITE_API_BASE ?? 'https://chat.lost.plus'
).replace(/\/+$/, '');

export interface DashConfig {
	apiBase: string;
	apiKey: string;
}

export class UnauthorizedError extends Error {
	constructor(message = 'Invalid API key') {
		super(message);
		this.name = 'UnauthorizedError';
	}
}

export function loadConfig(): DashConfig {
	let stored: Partial<DashConfig> = {};

	try {
		stored = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}') ?? {};
	} catch {
		stored = {};
	}

	return {
		apiBase: (stored.apiBase || DEFAULT_API_BASE).replace(/\/+$/, ''),
		apiKey: stored.apiKey ?? ''
	};
}

export function saveConfig(config: DashConfig): void {
	localStorage.setItem(
		STORAGE_KEY,
		JSON.stringify({ ...config, apiBase: config.apiBase.replace(/\/+$/, '') })
	);
}

export function clearApiKey(): void {
	const config = loadConfig();
	saveConfig({ ...config, apiKey: '' });
}

async function parseErrorMessage(response: Response): Promise<string> {
	try {
		const body = await response.json();
		const detail = body?.detail ?? body?.error?.message ?? body?.error;
		if (typeof detail === 'string' && detail) return detail;
	} catch {
		/* fall through to the status line */
	}
	return `${response.status} ${response.statusText}`.trim();
}

export async function apiFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
	const { apiBase, apiKey } = loadConfig();

	let response: Response;
	try {
		response = await fetch(`${apiBase}${path}`, {
			...options,
			headers: {
				'Content-Type': 'application/json',
				...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
				...(options.headers as Record<string, string> | undefined)
			}
		});
	} catch {
		// A blocked cross-origin request and an unreachable host are
		// indistinguishable from here — the browser hides the reason on purpose.
		throw new Error(
			`Could not reach ${apiBase}. Check that it is up and that it allows this origin.`
		);
	}

	if (response.status === 401 || response.status === 403) {
		throw new UnauthorizedError(await parseErrorMessage(response));
	}

	if (!response.ok) {
		throw new Error(await parseErrorMessage(response));
	}

	return response.json() as Promise<T>;
}
