#!/bin/bash

# Default values
SERVER_IP="10.172.13.12"
DURATION=30
PORT=5201
OUTPUT_FILE="test.json"

# Help function
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -c SERVER_IP   Server IP address (default: $SERVER_IP)"
    echo "  -t DURATION    Test duration in seconds (default: $DURATION)"
    echo "  -p PORT        Server port number (default: $PORT)"
    echo "  -J OUTPUT_FILE Output JSON file name (default: $OUTPUT_FILE)"
    echo "  -h             Show this help message"
}

# Parse command line options
while getopts "c:t:p:J:h" opt; do
    case $opt in
        c) SERVER_IP="$OPTARG" ;;
        t) DURATION="$OPTARG" ;;
        p) PORT="$OPTARG" ;;
        J) OUTPUT_FILE="$OPTARG" ;;
        h) show_help
           exit 0 ;;
        *) show_help
           exit 1 ;;
    esac
done

# Run iperf3 with the specified (or default) parameters
iperf3 -c "$SERVER_IP" \
       -t "$DURATION" \
       -p "$PORT" \
       -J "$OUTPUT_FILE"