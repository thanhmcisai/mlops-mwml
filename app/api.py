from http import HTTPStatus
from functools import wraps
from datetime import datetime
from typing import Dict, Any, Tuple, Callable

from fastapi import FastAPI, Request

# Define application
app = FastAPI(
    title="MLOps - Start to MLOps",
    description="Classify machine learning projects.",
    version="0.1",
)


def construct_response(func: Callable[..., Any]) -> Callable[..., Dict[str, Any]]:
    """Construct a JSON response for an endpoint."""

    @wraps(func)
    def wrap(request: Request, *args: Tuple[Any], **kwargs: Dict[Any, Any]) -> Dict[str, Any]:
        results: Any = func(request, *args, **kwargs)
        response: Dict[str, Any] = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,  # type: ignore
        }
        if "data" in results:
            response["data"] = results["data"]
        return response

    return wrap


@app.get('/')
@construct_response
def index(request: Request) -> Dict[str, Any]:
    """Health check"""
    response: Dict[str, Any] = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response
