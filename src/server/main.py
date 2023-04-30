import json
import traceback
from pathlib import Path

import uvicorn
from description import api_description, predict_desc
from fastapi import FastAPI, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, RedirectResponse
from server_logger import logger

import src.server.router as router
from src.server.config_server import get_server_args
from src.server.server_store import server_store

# include project path so that .env file can be read from all locations


def catch_exceptions_middleware(request, call_next):
    try:
        return call_next(request)
    except Exception as err:
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=jsonable_encoder(
                {
                    "message": "Internal server error. Check console log for more information."
                }
            ),
        )


tags_metadata = [
    {
        "name": "available models",
        "description": "All available models from the `MODEL_DIRECTORY` directory defined in the `.env` file",
    },
    {
        "name": "predict",
        "description": predict_desc,
    },
]

app = FastAPI(debug=True, openapi_tags=tags_metadata, description=api_description)
app.middleware("http")(catch_exceptions_middleware)


@app.on_event("shutdown")
def shutdown_event():
    logger.info("Application shutdown")


@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url="/docs")


app.include_router(router.router)

with open(Path(Path(__file__).parent.resolve(), "openapi_spec.json"), "w+") as file:
    file.write(json.dumps(app.openapi()))
    file.close()

if __name__ == "__main__":
    args, config, _ = get_server_args()
    server_store.set_config(config, args)
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        loop="asyncio",
        reload=args.hot_reload,
    )
