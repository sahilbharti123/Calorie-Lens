/** Stands in for the native-dependent API client. See tests/register-alias.mjs. */
export class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(status: number, detail: unknown) {
    super(String(detail));
    this.status = status;
    this.detail = detail;
  }
}

/** No base URL configured means the remote branch is never taken. */
export function apiUrl() {
  return '';
}

export async function apiRequest<T>(): Promise<T> {
  throw new ApiError(0, 'the network is not reachable from a unit test');
}
