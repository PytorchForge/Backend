import uvicorn
def main():
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
        workers=1,
        lifespan="on",
        loop="auto"
    )
    
if __name__ == "__main__":
    main()