/**
 * Talking to Grimoire.
 *
 * The page is served by an nginx that proxies /stats through to the gateway and
 * attaches the API key on the way. So these are ordinary same-origin requests:
 * no credential is held in the browser, no CORS is involved, and there is
 * nothing to sign in to. Whatever guards this site is what guards the data.
 */

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
	let response: Response;
	try {
		response = await fetch(path, {
			...options,
			headers: {
				'Content-Type': 'application/json',
				...(options.headers as Record<string, string> | undefined)
			}
		});
	} catch {
		throw new Error('Could not reach the gateway.');
	}

	if (!response.ok) {
		throw new Error(await parseErrorMessage(response));
	}

	return response.json() as Promise<T>;
}
