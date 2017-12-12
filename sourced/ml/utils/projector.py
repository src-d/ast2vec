import logging
from http.server import HTTPServer, SimpleHTTPRequestHandler, test
import os
import shutil
import threading
import time


class CORSWebServer(object):
    def __init__(self):
        self.thread = None
        self.server = None

    def serve(self):
        outer = self

        class ClojureServer(HTTPServer):
            def __init__(self, *args, **kwargs):
                HTTPServer.__init__(self, *args, **kwargs)
                outer.server = self

        class CORSRequestHandler(SimpleHTTPRequestHandler):
            def end_headers(self):
                self.send_header("Access-Control-Allow-Origin", "*")
                SimpleHTTPRequestHandler.end_headers(self)

        test(CORSRequestHandler, ClojureServer)

    def start(self):
        self.thread = threading.Thread(target=self.serve)
        self.thread.start()

    def stop(self):
        if self.running:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join()
            self.server = None
            self.thread = None

    @property
    def running(self):
        return self.server is not None


web_server = CORSWebServer()


def present_embeddings(destdir, run_server, labels, index, embeddings):
    log = logging.getLogger("projector")
    log.info("Writing Tensorflow Projector files...")
    if not os.path.isdir(destdir):
        os.makedirs(destdir)
    os.chdir(destdir)
    metaf = "id2vec_meta.tsv"
    with open(metaf, "w") as fout:
        if len(labels) > 1:
            fout.write("\t".join(labels) + "\n")
        for item in index:
            if len(labels) > 1:
                fout.write("\t".join(item) + "\n")
            else:
                fout.write(item + "\n")
        log.info("Wrote %s", metaf)
    dataf = "id2vec_data.tsv"
    with open(dataf, "w") as fout:
        for vec in embeddings:
            fout.write("\t".join(str(v) for v in vec))
            fout.write("\n")
        log.info("Wrote %s", dataf)
    jsonf = "id2vec.json"
    with open(jsonf, "w") as fout:
        fout.write("""{
  "embeddings": [
    {
      "tensorName": "id2vec",
      "tensorShape": [%s, %s],
      "tensorPath": "http://0.0.0.0:8000/%s",
      "metadataPath": "http://0.0.0.0:8000/%s"
    }
  ]
}
""" % (len(embeddings), len(embeddings[0]), dataf, metaf))
    log.info("Wrote %s", jsonf)
    if run_server and not web_server.running:
        web_server.start()
    url = "http://projector.tensorflow.org/?config=http://0.0.0.0:8000/" + jsonf
    log.info(url)
    if run_server:
        if shutil.which("xdg-open") is not None:
            os.system("xdg-open " + url)
        else:
            browser = os.getenv("BROWSER", "")
            if browser:
                os.system(browser + " " + url)
            else:
                print("\t" + url)


def wait():
    log = logging.getLogger("projector")
    secs = int(os.getenv("PROJECTOR_SERVER_TIME", "60"))
    log.info("Sleeping for %d seconds, safe to Ctrl-C" % secs)
    try:
        time.sleep(secs)
    except KeyboardInterrupt:
        pass
    web_server.stop()
