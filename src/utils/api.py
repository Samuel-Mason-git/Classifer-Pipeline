import requests
from typing import Optional, Dict, Any


def call_api(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    debug: bool = False,
    ) -> Dict[str, Any]:
    """
    Generic GET request helper.

    Args:
        url: API endpoint
        headers: request headers (e.g. auth)
        params: query parameters
        debug: print request/response info

    Returns:
        Parsed JSON response
    """
    try:
        if debug:
            print(f"[DEBUG] Requesting: {url}")
            print(f"[DEBUG] Params: {params}")
            print(f"[DEBUG] Headers: {headers}")

        response = requests.get(url, headers=headers, params=params, timeout=30)

        if debug:
            print(f"[DEBUG] Status Code: {response.status_code}")

        response.raise_for_status()

        data = response.json()

        if debug:
            # print small sample, not full response
            if isinstance(data, dict):
                print(f"[DEBUG] Response keys: {list(data.keys())}")
            else:
                print(f"[DEBUG] Response type: {type(data)}")

        return data

    except requests.RequestException as e:
        raise RuntimeError(
            f"API request failed: {url} | params={params}"
        ) from e