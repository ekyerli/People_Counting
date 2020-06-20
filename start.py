from tkinter import Tk, filedialog, Button, Label
import people_counter as pc
import datetime
import pl

class Movie:
    filenameOpen = ""
    filenameSave = ""

    def openFile(self):
        self.filenameOpen = filedialog.askopenfilename(initialdir="/", title="Select file",
                                                       filetypes=(("Video files", "*.mp4"), ("all files", "*.*")))
        print(self.filenameOpen)

    def saveFile(self):
        self.filenameSave = filedialog.asksaveasfilename(initialdir="/", title="Select file",
                                                         filetypes=(("Video files", "*.avi"), ("all files", "*.*")))
        print(self.filenameSave)

    def start(self):
        info, stat = pc.counter(self.filenameOpen, self.filenameSave)
        currentDate = datetime.datetime.today()
        pl.graph(stat)


def main():
    root = Tk()
    root.title("Ege Üniversitesi Bitirme Tezi(Caner YILDIRIM-Emir Kaan YERLİ)")
    root.minsize(width=500, height=360)

    obj = Movie()

    button1 = Button(root, text="YÜKLE !", command=obj.openFile)
    button1.place(x=150, y=150)

    button2 = Button(root, text="KAYDET !", command=obj.saveFile)
    button2.place(x=220, y=150)

    button3 = Button(root, text="SAY !", command=obj.start)
    button3.place(x=300, y=150)
    #button3.place(x=150, y=10)

    root.mainloop()


if __name__ == "__main__":
    main()
