from customtkinter import *
from customtkinter import filedialog, CTkFrame
import numpy as np
from analysis6mzip import unzipper
import pandas as pd
from zipfile import ZipFile
# import tkinter as tk
# from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib

SIZE = (1000,600)
matplotlib.use('TkAgg')

# matplotlib.rcParams.update({# Use mathtext, not LaTeX
#                             'text.usetex': False,
#                             # Use the Computer modern font
#                             'font.family': 'serif',
#                             'font.serif': 'cmr10',
#                             'mathtext.fontset': 'cm',
#                             # Use ASCII minus
#                             'axes.unicode_minus': False,
#                             })

lod = True
app = CTk()
app.resizable(width=False, height=False)
app.title("GEM Imaging")
app.geometry(f'{SIZE[0]}x{SIZE[1]}')

def toggle_theme():
    global lod
    if lod:
        lod_btn.configure(text = 'Dark Mode')
        set_appearance_mode("light")
        lod = False
        return None
    lod_btn.configure(text = 'Light Mode')
    lod = True
    set_appearance_mode('dark')
    return

def plotFile(data,i=0):
    fig = Figure(figsize=(9,7))
    ax = fig.add_subplot(111)
    ax.set_xlabel('samples')
    ax.set_ylabel('charge')
    ax.plot(data.iloc[i])
    if str(i+1).endswith('1'):
        ax.set_title(f"{i+1}st strip")
    elif str(i+1).endswith('2'):
        ax.set_title(f"{i+1}nd strip")
    elif str(i+1).endswith('3'):
        ax.set_title(f"{i+1}rd strip")
    else:
        ax.set_title(f"{i+1}th strip")
    if str(i+1)=='12':
            ax.set_title(f"{i+1}th strip")
    elif str(i+1)=='11':
            ax.set_title(f"{i+1}th strip")
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=0,column=0)

    	
    toolbar = NavigationToolbar2Tk(canvas, plot_frame, pack_toolbar=False)
    toolbar.update()
    toolbar.grid(row = 1,column= 0,sticky='we')


    

def selectFile():
    filetypes = [
        ('Compressed files','*.zip'),
        ('text files',('*.dat',' *.csv')),        
        ('All files','*.*')
	]
    wind = filedialog.askopenfile(filetypes= filetypes)
    global data_file
    file_label.configure(text= wind.name,wraplength = 170)
    stp_num = strip_num_btn.get()
    print(stp_num)
    print(wind)
    if '.zip' in wind.name:
        data_file = unzipper(wind.name)
    else:
        data = wind.name
        data_file = pd.read_csv(data,header=None)
    plotFile(data_file,i = int(stp_num))
    
def UpdateGraph(i):
    i = strip_num_btn.get()
    try:
        print(data_file)
        for wids in plot_frame.winfo_children():
            wids.destroy()
        
        plotFile(data_file,i = int(i)-1)
    except NameError:
         print('input file missing')
         


input_controls = CTkFrame(app,fg_color='red',width = SIZE[0]/3,height=SIZE[1])
input_controls.grid(row=0,column=1,sticky ='nsew')
# CTkLabel(input_controls,fg_color= 'red').pack(expand=True, fill='both')

plot_frame = CTkFrame(app, fg_color='blue',width = 2*SIZE[0]/3,height=SIZE[1])
plot_frame.grid(row=0,column=0,sticky ='nsew')


file_btn = CTkButton(master=input_controls, text = "Select File",corner_radius=0, command=selectFile)
file_btn.pack(anchor='center', expand=True, padx=10,pady= 20)#.place(relx = .5, rely=0.5, anchor = "center")
file_label =  CTkLabel(input_controls, text='No file selected',fg_color= 'red')
file_label.pack(expand=True, fill='both')

strip_num_btn = CTkComboBox(master=input_controls, values=[str(i) for i in range(1,257)],command=UpdateGraph)
strip_num_btn.pack(anchor='center', expand=True, padx=10,pady= 20)#place(relx=.5, rely=0.7, anchor = 'center')

strip_num_btn.bind('<Return>',UpdateGraph)

# strip_upd_btn = CTkButton(master=input_controls,text='Update', command=UpdateGraph)
# strip_upd_btn.pack(anchor='center', expand=True, padx=10,pady= 20)#place(relx=.5, rely=0.7, anchor = 'center')

lod_btn = CTkButton(master=input_controls, text = "Toggle Light and Dark mode",corner_radius=0, command=toggle_theme)
lod_btn.pack(anchor='center', expand=True, padx=10,pady= 20)#.place(relx = .8, rely=0.7, anchor = "center")

app.mainloop()