import sys
import time

def main():
    delay = 0.1  # Default delay of 1/10th second

    if len(sys.argv) > 1:
        try:
            delay = float(sys.argv[1])
        except ValueError:
            pass

    for line in sys.stdin:
        # Print the line to stdout
        sys.stdout.write(line)
        sys.stdout.flush()
        
        # Checking for quit command to stop delaying (optional, but keep consistent behavior)
        if line.strip() == ":quit":
            break
            
        time.sleep(delay)

if __name__ == "__main__":
    main()
