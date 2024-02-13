import os
import glob
from typing import Tuple
import customtkinter as ctk
from customtkinter import filedialog, CTkFrame
import numpy as np
from analysis6mzip import unzipper
import pandas as pd
# import tkinter as tk
# from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import matplotlib
from utils import to_inch


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





class SettingsFrame(CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.lod_btn = ctk.CTkButton(master=self, text = "Light mode",corner_radius=0, command=self.toggle_theme)
        self.lod_btn.pack(anchor='center', expand=True, padx=10,pady= 20)#.place(relx = .8, rely=0.7, anchor = "center")

        self.lod = True
    def toggle_theme(self):
        if self.lod:
            self.lod_btn.configure(text = 'Dark Mode')
            ctk.set_appearance_mode("light")
            self.lod = False
            return None
        self.lod_btn.configure(text = 'Light Mode')
        self.lod = True
        ctk.set_appearance_mode('dark')
        return None


class Tab1Frame(CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.graphframe = CTkFrame(master = self, width=2*SIZE[0]/3, height=SIZE[1],fg_color='blue')
        self.inputframe = CTkFrame(master=self,width  = SIZE[0]/3, height = SIZE[1], fg_color = 'red')
        # self.graphframe.grid(row = 0,column=0)
        # self.inputframe.grid(row = 0, column = 1)
        self.graphframe.pack(expand=True, fill = 'both',side='left')
        self.inputframe.pack(expand=True, fill = 'both',side='left')

        self.file_btn = ctk.CTkButton(master=self.inputframe, text = "Select File",corner_radius=0, command=self.selectFile)
        self.file_btn.pack(anchor='center', expand=True, padx=10,pady= 20)#.place(relx = .5, rely=0.5, anchor = "center")
        self.file_label =  ctk.CTkLabel(master=self.inputframe, text='No file selected',fg_color= 'red')
        self.file_label.pack(expand=True, fill='both')

        self.strip_num_btn = ctk.CTkComboBox(master=self.inputframe, values=[str(i) for i in range(1,257)],command=self.UpdateGraph)
        self.strip_num_btn.pack(anchor='center', expand=True, padx=10,pady= 20)#place(relx=.5, rely=0.7, anchor = 'center')
        self.strip_num_btn.bind('<Return>',self.UpdateGraph)

    def plotFile(self,data,i=0,peaks= False):
        fig = Figure(figsize=(6.3,6))
        ax = fig.add_subplot(111)
        ax.set_xlabel('samples')
        ax.set_ylabel('charge')
        di =.125-data.iloc[i]
        ax.plot(di)
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
        return fig
    
    def setplot(self,fig):
        canvas = FigureCanvasTkAgg(fig, master=self.graphframe)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=0,column=0)    	
        toolbar = NavigationToolbar2Tk(canvas, self.graphframe, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row = 1,column= 0,sticky='we')
    
    def UpdateGraph(self,i):
        #i = self.strip_num_btn.get()
        try:

            for wids in self.graphframe.winfo_children():
                wids.destroy()
            
            self.setplot(self.plotFile(data_file,i = int(i)-1))
        except NameError:
            print('input file missing')
    
    def selectFile(self):
        filetypes = [
            ('Compressed files','*.zip'),
            ('text files',('*.dat',' *.csv')),        
            ('All files','*.*')
        ]
        wind = filedialog.askopenfile(filetypes= filetypes)
        global data_file
        self.file_label.configure(text= wind.name,wraplength = 170)
        stp_num = self.strip_num_btn.get()
        if '.zip' in wind.name:
            data_file = unzipper(wind.name)
        else:
            data = wind.name
            data_file = pd.read_csv(data,header=None)
        self.setplot(self.plotFile(data_file,i = int(stp_num)))
    

class PeakView(CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.plotframe = CTkFrame(master=self)
        self.buttonframe = CTkFrame(master=self)

        # self.plotframe.pack(expand=True, fill = 'both',side='left')
        # self.buttonframe.pack(expand=True, fill = 'both',side='left')

        self.plotframe.grid(row=0,column = 0,columnspan =3,sticky='snew')
        self.buttonframe.grid(row=0,column = 3,sticky='snew')

        self.peak_det_btn = ctk.CTkButton(master=self.buttonframe, text='detect peaks',command=self.detectpeaks)
        self.peak_det_btn.grid(row= 0,column= 0, padx=10,pady= 20,sticky='ew',columnspan=2)
        # self.peak_det_btn.grid (row= 0,column= 0, padx=10,pady= 20,sticky='ew')
        #self.peak_det_btn.pack(anchor='center', expand=True, padx=10,pady= 20)

        self.prev_button = ctk.CTkButton(master=self.buttonframe,text='<',command=self.prev_graph)
        self.prev_button.grid(row=1,column=0, padx=10,pady= 20)
        #self.prev_button.pack(anchor='center', expand=True, padx=10,pady= 20)

        self.next_button = ctk.CTkButton(master=self.buttonframe,text='>',command=self.next_graph)
        self.next_button.grid(row=1,column=1, padx=10,pady= 20)
        #self.next_button.pack(anchor='center', expand=True, padx=10,pady= 20)

        self.save_btn = ctk.CTkButton(master=self.buttonframe, text='save',command=self.save)
        self.save_btn.grid(row=2,column=0, padx=10,pady= 20)
        #self.save_btn.pack(anchor='center', expand=True, padx=10,pady= 20)

        self.save_all_btn = ctk.CTkButton(master=self.buttonframe, text='save all',command=self.saveall)
        self.save_all_btn.grid(row=2,column=1, padx=10,pady= 20)
        # self.save_all_btn.pack(anchor='center', expand=True, padx=10,pady= 20)

        self.next_file= None
        self.prev_file = None
        self.current_file = 0
        self.data_files_list=None
        self.files_list_len = None
        self.directory = None
        
    def getfiles(self):
        self.directory = filedialog.askdirectory()
        self.data_files_list= {i:k for i,k in enumerate(glob.glob(os.path.join(self.directory,'*.zip')))}
        self.files_list_len = len(self.data_files_list)

    def setplot(self,fig):
        plt.close()
        for wids in self.plotframe.winfo_children():
                wids.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.plotframe)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=0,column=0)    	
        toolbar = NavigationToolbar2Tk(canvas, self.plotframe, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row = 1,column= 0,sticky='we')
    
    def detectpeaks(self):
        self.getfiles()
        data= unzipper(self.data_files_list[self.current_file])

        
        from scipy.signal import find_peaks
        data_mins =.125- data.min(axis=1)
        x_min = data_mins.iloc[:128]
        y_min = data_mins.iloc[128:]   

        xp,xh = find_peaks(x_min,height=0.075)
        yp,yh = find_peaks(y_min,height=0.075)

        fig,(ax1,ax2) = plt.subplots(nrows=2, ncols=1)
        fig.suptitle(f'{self.data_files_list[self.current_file]}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('charge')

        ax1.plot(xp,xh['peak_heights'],linestyle = '',marker='.')
        ax2.plot(yp,yh['peak_heights'],linestyle = '',marker='.')
        ax2.set_xlabel('y')
        ax2.set_ylabel('charge')
        self.setplot(fig)



    def next_graph(self):
        if self.current_file<self.files_list_len:

            self.current_file = self.current_file+1
            data= unzipper(self.data_files_list[self.current_file])        
            from scipy.signal import find_peaks
            data_mins =.125- data.min(axis=1)
            x_min = data_mins.iloc[:128]
            y_min = data_mins.iloc[128:]   

            xp,xh = find_peaks(x_min,height=0.075)
            yp,yh = find_peaks(y_min,height=0.075)

            fig,(ax1,ax2) = plt.subplots(nrows=2, ncols=1)

            fig.suptitle(f'{self.data_files_list[self.current_file]}')

            ax1.plot(xp,xh['peak_heights'],linestyle = '',marker='.')
            ax1.set_xlabel('x')
            ax1.set_ylabel('charge')
            ax2.plot(yp,yh['peak_heights'],linestyle = '',marker='.')
            ax2.set_xlabel('y')
            ax2.set_ylabel('charge')
            self.setplot(fig)

    def prev_graph(self):
        if self.current_file>0:
            self.current_file = self.current_file-1
            data= unzipper(self.data_files_list[self.current_file])        
            from scipy.signal import find_peaks
            data_mins =.125- data.min(axis=1)
            x_min = data_mins.iloc[:128]
            y_min = data_mins.iloc[128:]   

            xp,xh = find_peaks(x_min,height=0.075)
            yp,yh = find_peaks(y_min,height=0.075)

            fig,(ax1,ax2) = plt.subplots(nrows=2, ncols=1)
            fig.suptitle(f'{self.data_files_list[self.current_file]}')

            ax1.plot(xp,xh['peak_heights'],linestyle = '',marker='.')
            ax1.set_xlabel('x')
            ax1.set_ylabel('charge')
            ax2.plot(yp,yh['peak_heights'],linestyle = '',marker='.')
            ax2.set_xlabel('y')
            ax2.set_ylabel('charge')
            self.setplot(fig)

    def save(self):
        pass

    def saveall(self):
        pass

class TabWrapper(ctk.CTkTabview):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.add('Tab')
        self.tab1 = Tab1Frame(self.tab('Tab'))
        self.tab1.pack(expand=True, fill = 'both')

        self.add('tab2')
        self.peakanalysis = PeakView(self.tab('tab2'))
        self.peakanalysis.pack(expand=True,fill='both')
        self.add('settings')
        self.settings = SettingsFrame(self.tab('settings'))
        self.settings.pack(expand=True,fill='both')



class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("GEM Imaging")
        self.geometry(f'{SIZE[0]}x{SIZE[1]}')
        self.iconbitmap()
        self.view = TabWrapper(self)
        self.view.pack(expand=True, fill = 'both')
        #self.resizable(width=False, height=False)
app = App()
app.mainloop()