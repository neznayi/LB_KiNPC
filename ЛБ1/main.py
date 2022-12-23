import time
import pywinauto
from pywinauto.application import Application
app=Application().start(cmd_line=r"D:\PuTTY\putty.exe")
time.sleep(5)
app=Application().connect(title="PuTTY Configuration")
window=app.PuTTYConfigBox
window.set_focus()
window[u"Host Name (or IP address):Edit"].type_keys("tty.sdf.org")
window["Open"].click()
