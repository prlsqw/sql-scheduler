import sys
import time

def main():
    delay = 0.1  # delay of 1/10th second

    if len(sys.argv) > 1:
        try:
            delay = float(sys.argv[1])
        except ValueError:
            pass

    for line in sys.stdin:
        sys.stdout.write(line)
        sys.stdout.flush()
        
        if line.strip() == ":quit":
            break
            
        time.sleep(delay)

if __name__ == "__main__":
    main()
