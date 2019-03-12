from tkinter import *
from tkinter.filedialog import *
from PIL import Image, ImageTk
from tkinter import messagebox
import os, csv



if __name__ == "__main__":
    root = Tk()
    global count, x1, y1
    count = 0
    #setting up a tkinter canvas with scrollbars
    frame = Frame(root, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    xscroll = Scrollbar(frame, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E+W)
    yscroll = Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=N+S)
    canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    canvas.grid(row=0, column=0, sticky=N+S+E+W)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill=BOTH,expand=1)

    
    
	#path = askdirectory(parent=root, initialdir="C:/Users",title='Select where to save')
    filename = askopenfilename(parent=root, initialdir="C:/Users",title="Select csv to save in")
	
    #fullpath = ''join(path, filename)
	
    #adding the image
    File = askopenfilename(parent=root, initialdir="C:/Users/Psy/Downloads/Data/Elephants",title='Select file')
    img = ImageTk.PhotoImage(Image.open(File))
    canvas.create_image(0,0,image=img,anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))

    #function to be called when mouse is clicked
    def printcoords(event):
        global count, x1, y1
		#vai buscar o nome da imagem sem o .jpg
        #print (os.path.splitext(os.path.basename(File))[0])
        #outputting x and y coords to console
        if messagebox.askokcancel("Python",str(event.x) + "," + str(event.y)):
            if count == 0:
                #save to csv
                x1=event.x
                y1=event.y
            count += 1
            print (count)
            if count == 2:
                row = [os.path.splitext(os.path.basename(File))[0],'Elephant',str(x1),str(y1),str(event.x),str(event.y)]
                with open(filename,'a+', newline ='') as fd:
                    writer = csv.writer(fd)
                    writer.writerow(row)
                root.destroy()		
    #mouseclick event
    canvas.bind("<Button 1>",printcoords)

    root.mainloop()