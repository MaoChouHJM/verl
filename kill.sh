ps -aux | grep Run | awk '{print $2}' | xargs kill -9
