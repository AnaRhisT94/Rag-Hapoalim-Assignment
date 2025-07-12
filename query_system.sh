 # Query the system
   curl -X POST "http://localhost:8000/query" \
        -H "Content-Type: application/json" \
        -d '{"query": "Your question here", "top_k": 3}'

   # Check status
   curl "http://localhost:8000/health"
   curl "http://localhost:8000/index-status"

   # Rebuild index
   curl -X POST "http://localhost:8000/rebuild-index"