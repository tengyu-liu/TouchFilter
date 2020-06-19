from tkinter import *

class TCFrame(Frame):
  def __init__(self, dl, gl, master=None):
    Frame.__init__(self, master)
    self.master = master
    self.dl = dl
    self.gl = gl
    self.init_window()
  
  def init_window(self):
    self.master.title("Train Controller v0.1")
    self.pack(fill=BOTH, expand=1)
    self.command_field = Entry(self)
    self.command_field.pack()
    self.submit_button = Button(self, text="Submit", command=self.fun_submit)
    self.submit_button.pack()
    self.dl_label = Label(self, text='dl=%e'%self.dl)
    self.dl_label.pack()
    self.gl_label = Label(self, text='gl=%e'%self.gl)
    self.gl_label.pack()

  def fun_submit(self):
    command = self.command_field.get()
    try:
      parse = command.split('=')
      if parse[0].strip().lower() == 'dl':
        self.dl = float(parse[1].strip().lower())
        print('\nSetting descriptor learning rate to: %e'%self.dl)
      elif parse[0].strip().lower() == 'gl':
        self.gl = float(parse[1].strip().lower())
        print('\nSetting generator learning rate to: %e'%self.gl)
      else:
        raise NotImplemented
    except:
      print('Error parsing command "%s"'%command)
    self.command_field.delete(0, len(command))
    self.dl_label.config(text="dl=%e"%self.dl)
    self.gl_label.config(text="gl=%e"%self.gl)

class TrainController():
  def __init__(self, dl, gl):
    self.tk = Tk()
    self.tk.geometry("400x300")
    self.app = TCFrame(dl, gl, self.tk)
    self.tk.mainloop()
  
  def get_dl(self):
    return self.app.dl
  
  def get_gl(self):
    return self.app.gl
     
if __name__ == "__main__":
    TrainController(0, 0)
