import queue

addresses = ['https://www.google.com', 'https://www.google.com', 'https://www.google.com']
# Create queue and add addresses
q = queue.Queue()
for url in addresses:
    q.put(url)
