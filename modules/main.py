import uvicorn
def main():
    uvicorn.run(
        "server:app",
        host="http://localhost",
        port=8000,
        reload=True,
        log_level="info",
        workers=1,
        factory=True,
        lifespan="on",
        loop="auto"
    )
    
if __name__ == "__main__":
    main()