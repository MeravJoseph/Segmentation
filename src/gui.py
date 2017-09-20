# *************************************************
# *** Semantic Segmentation from Training Image ***
# ************ Merav Joseph 200652063 *************
# ************* Shir Amir 209712801 ***************
# *************************************************
import PIL.Image
import PIL.ImageTk
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
import cv2
import segment #our segmentation module

def get_train():
    """ callback function """
    train_img_path.set(askopenfilename())

def get_segment():
    """ callback function """
    labels_img_path.set(askopenfilename())

def get_new():
    """ callback function """
    test_img_path.set(askopenfilename())

def get_output():
    """ callback function """
    output_dir.set(askdirectory())

def compute():
    """ callback function """
    # Prepare parameters
    kwargs = {}
    kwargs['TRAIN_IMG_PATH'] = train_img_path.get()
    kwargs['LABELS_IMG_PATH'] = labels_img_path.get()
    kwargs['TEST_IMG_PATH'] = test_img_path.get()
    kwargs['OUTPUT_DIR'] = output_dir.get()
    kwargs['FRAG_AMOUNT'] = frag_amount.get()
    kwargs['PATCH_SIZE'] = patch_size.get()
    kwargs['GRABCUT_THRESH'] = grabcut_thresh.get()
    kwargs['GRABCUT_ITER'] = grabcut_iter.get()
    kwargs['SLIC_SIGMA'] = slic_sigma.get()

    # Compute segmentation
    res_path = segment.segment_image(**kwargs)
    res = PIL.Image.open(res_path)
    res = PIL.ImageTk.PhotoImage(res)
    # image = Label(canvas, image=res, anchor=CENTER)
    image.configure(image=res)
    image.image = res
    image.grid(row=2, column=2, columnspan=3, rowspan=12)

root = Tk()
root.title("Segmentation2")
root.configure(background='#ccffcc')
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))

root.focus_set() # <-- move focus to this widget
root.bind("<Escape>", lambda e: e.widget.quit())

# Define all the variables
train_img_path = StringVar()
labels_img_path = StringVar()
test_img_path = StringVar()
output_dir = StringVar()
frag_amount = StringVar()
patch_size = StringVar()
grabcut_thresh = StringVar()
grabcut_iter = StringVar()
slic_sigma = StringVar()

# Header
headline = Label(root, font="Gisha 20 bold", bg='#ccffcc', fg='#006600', text="Segmentation App", anchor=CENTER)
headline.grid(row=0, column=0, columnspan=7, padx=10, pady=10)

# Left input and output settings
btn1 = Button(root, font="Gisha 12", fg='#006600', bg='#b3ffb3', command=get_train, text="Train Image", width=12)
btn1.grid(row=1, column=0, padx=15, pady=0)
train_img_path.set("../images/giraffes_train.jpg")
btn2 = Button(root, font="Gisha 12", fg='#006600', bg='#b3ffb3', command=get_segment, text="Labeled Image", width=12)
btn2.grid(row=2, column=0, padx=15, pady=0)
labels_img_path.set("../images/giraffes_train_labels.tif")
btn3 = Button(root, font="Gisha 12", fg='#006600', bg='#b3ffb3', command=get_new, text="Test Image", width=12)
btn3.grid(row=3, column=0, padx=15, pady=0)
test_img_path.set("../images/giraffes_test.jpg")
btn4 = Button(root, font="Gisha 12", fg='#006600', bg='#b3ffb3', command=get_output, text="Output Directory", width=12)
btn4.grid(row=4, column=0, padx=15, pady=0)
output_dir.set("../results/")
btn5 = Button(root, font="Gisha 12 bold", fg='#006600', bg='#b3ffb3', command=compute, text="GO!", width=11)
btn5.grid(row=5, column=0, padx=15, pady=0)

# Main frame for showing results
canvas = Canvas(relief=RIDGE, bd=5, width=0.75*w, height=0.75*h).grid(row=1, column=1, columnspan=5, rowspan=22, sticky=E)
image = Label(canvas, image="", anchor=CENTER)

# Right parameter settings
label1 = Label(root, font="Gisha 12", fg='#006600', bg='#ccffcc', text="Amount \n of Fragments:", width=15)
label1.grid(row=1, column=6, padx=0, pady=0)
entry1 = Entry(root, font="Gisha 12", fg='#006600', textvariable=frag_amount, width=12)
entry1.grid(row=2, column=6, padx=15, pady=0)
frag_amount.set("400")

label2 = Label(root, font="Gisha 12", fg='#006600', bg='#ccffcc', text="Patch Size:", width=12)
label2.grid(row=3, column=6, padx=15, pady=0)
entry2 = Entry(root, font="Gisha 12", fg='#006600', textvariable=patch_size, width=12)
entry2.grid(row=4, column=6, padx=15, pady=0)
patch_size.set("9")

label3 = Label(root, font="Gisha 12", fg='#006600', bg='#ccffcc', text="Grabcut \n Threshold:", width=12)
label3.grid(row=5, column=6, padx=0, pady=0)
entry3 = Entry(root, font="Gisha 12", fg='#006600', textvariable=grabcut_thresh, width=12)
entry3.grid(row=6, column=6, padx=15, pady=0)
grabcut_thresh.set("0.0001")

label4 = Label(root, font="Gisha 12", fg='#006600', bg='#ccffcc', text="Grabcut \n Iterations:", width=12)
label4.grid(row=7, column=6, padx=0, pady=0)
entry4 = Entry(root, font="Gisha 12", fg='#006600', textvariable=grabcut_iter, width=12)
entry4.grid(row=8, column=6, padx=15, pady=0)
grabcut_iter.set("10")

label5 = Label(root, font="Gisha 12", fg='#006600', bg='#ccffcc', text="SLIC Sigma:", width=12)
label5.grid(row=9, column=6, padx=0, pady=0)
entry5 = Entry(root, font="Gisha 12", fg='#006600', textvariable=slic_sigma, width=12)
entry5.grid(row=10, column=6, padx=15, pady=0)
slic_sigma.set("5")

explanations = "Press Esc to leave \n Press GO! once and be patient :)"
footer = Label(root, font="Gisha 14", fg='#006600', bg='#ccffcc', text=explanations, anchor=CENTER)
footer.grid(row=20, column=0, columnspan=7, padx=10, pady=10)

root.mainloop()