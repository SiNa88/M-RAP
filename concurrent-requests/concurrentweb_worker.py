class Worker(Thread):
    def __init__(self, request_queue):
        Thread.__init__(self)
        self.queue = request_queue
        self.results = []

    def run(self):
        while True:
            content = self.queue.get()
            if content == "":
                break
            request = urllib.request.Request(content)
            response = urllib.request.urlopen(request)
            self.results.append(response.read())
            self.queue.task_done()
