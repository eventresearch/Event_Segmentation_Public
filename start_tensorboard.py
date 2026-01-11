from tensorboard import program
import time

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', 'segmentation_results', '--host', '0.0.0.0', '--port', '6012'])
# tb.configure(argv=[None, '--logdir', 'segmentation_results', '--host', '0.0.0.0', '--port', '6012'])
url = tb.launch()
print(f"TensorBoard URL: {url}")

# Keep the process alive
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("TensorBoard stopped.")
