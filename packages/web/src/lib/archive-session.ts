import { toast } from "sonner";
import { browserApiFetch } from "@/lib/browser-api-fetch";

const GENERIC_ARCHIVE_ERROR = "Failed to archive session";

/**
 * Read the control plane's refusal reason so a queued-work or cancelled-session
 * block explains itself instead of surfacing the generic failure. Falls back to
 * the generic text for a non-JSON or empty body.
 */
async function archiveErrorMessage(response: Response): Promise<string> {
  try {
    const data: unknown = await response.json();
    if (
      data !== null &&
      typeof data === "object" &&
      "error" in data &&
      typeof (data as { error: unknown }).error === "string" &&
      (data as { error: string }).error.length > 0
    ) {
      return (data as { error: string }).error;
    }
  } catch {
    // Non-JSON body; fall through to the generic message.
  }
  return GENERIC_ARCHIVE_ERROR;
}

/**
 * Archives a session via the API.
 *
 * Returns `true` when the request succeeds. Callers are responsible for
 * updating any client-side caches or navigation state.
 */
export async function archiveSession(sessionId: string): Promise<boolean> {
  try {
    const response = await browserApiFetch(`/api/sessions/${sessionId}/archive`, {
      method: "POST",
    });
    if (!response.ok) {
      toast.error(await archiveErrorMessage(response));
      return false;
    }

    return true;
  } catch {
    toast.error(GENERIC_ARCHIVE_ERROR);
    return false;
  }
}
