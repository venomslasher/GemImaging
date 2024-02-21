import os
import glob
import customtkinter as ctk
from customtkinter import filedialog, CTkFrame
from tkinter import messagebox
import numpy as np
from analysis6mzip import unzipper , analysis6mzip,MAX_CHARGE, core_analyser
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import matplotlib
import seaborn as sns


SIZE = (1000,600)
matplotlib.use('TkAgg')

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
        fig ,ax= plt.subplots()#Figure(figsize=(6.3,6))
        #ax = fig.add_subplot(111)
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
        except TypeError:
            i = self.strip_num_btn.get()
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

    def EnterUpdateGraph(self,e):
        print(e)    

class PeakView(CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.plotframe = CTkFrame(master=self,width=2*SIZE[0]/3, height=SIZE[1])
        self.buttonframe = CTkFrame(master=self,width  = SIZE[0]/3, height = SIZE[1])

        self.plotframe.grid(row=0,column = 0,columnspan =3,sticky='nsew')
        self.buttonframe.grid(row=0,column = 3,sticky='nsew')

        self.peak_det_btn = ctk.CTkButton(master=self.buttonframe, text='detect peaks',command=self.detectpeaks)
        self.peak_det_btn.grid(row= 0,column= 0, padx=10,pady= 20,sticky='ew',columnspan=2)

        self.prev_button = ctk.CTkButton(master=self.buttonframe,text='<',command=self.prev_graph)
        self.prev_button.grid(row=1,column=0, padx=10,pady= 20)

        self.next_button = ctk.CTkButton(master=self.buttonframe,text='>',command=self.next_graph)
        self.next_button.grid(row=1,column=1, padx=10,pady= 20)

        self.save_btn = ctk.CTkButton(master=self.buttonframe, text='save',command=self.save)
        self.save_btn.grid(row=2,column=0, padx=10,pady= 20)

        self.save_all_btn = ctk.CTkButton(master=self.buttonframe, text='save all',command=self.saveall)
        self.save_all_btn.grid(row=2,column=1, padx=10,pady= 20)

        self.max_charge_label = ctk.CTkLabel(master=self.buttonframe,text= "Max charge")
        self.max_charge_label.grid(row=3,column=0, padx=1,pady= 2)

        self.max_charge = ctk.CTkEntry(master=self.buttonframe, placeholder_text= 0.125)
        self.max_charge.grid(row=3,column=1,columnspan=2, padx=1,pady= 2)

        self.height_thresh_label = ctk.CTkLabel(master=self.buttonframe,text= "Height threshold: ")
        self.height_thresh_label.grid(row=4,column=0, padx=1,pady= 2)

        self.height_thresh = ctk.CTkEntry(master=self.buttonframe, placeholder_text= 0.075)
        self.height_thresh.grid(row=4,column=1,columnspan=2, padx=1,pady= 2)

        self.width_thresh_label = ctk.CTkLabel(master=self.buttonframe,text= "Peak Width threshold(max) :")
        self.width_thresh_label.grid(row=5,column=0, padx=1,pady= 2)

        self.width_thresh = ctk.CTkEntry(master=self.buttonframe, placeholder_text= 8)
        self.width_thresh.grid(row=5,column=1,columnspan=2, padx=1,pady= 2)


        self.button_update = ctk.CTkButton(master=self.buttonframe, text='update',command=self.updategraph)
        self.button_update.grid(row=6,column=0, padx=10,pady= 20,columnspan=2)

        
        self.plotframe.bind('<KeyRelease-Left>', self.prev_graph)
        self.plotframe.bind('<KeyRelease-Right>', self.next_graph)
        self.plotframe.focus_set()

        self.next_file= None
        self.prev_file = None
        self.current_file = 0
        self.data_files_list=None
        self.files_list_len = None
        self.directory = None
        self.figure=None
        
    
    @staticmethod
    def make_pairs(xv,xh,yv,yh):
        x_list=[]
        y_list=[]
        ch_list= []
        for i,x in enumerate(xv):
            for j,y in enumerate(yv):
                xht = xh[i]
                yht = yh[j]
                if np.abs(xht-yht) < .01:
                    x_list.append(x)
                    y_list.append(y)
                    ch_list.append((xht+yht)/2)
        return pd.DataFrame({
            'x_loc':x_list,
            'y_loc':y_list,
            'charge':ch_list
        })

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

    
    def makeplot(self,data,max_charge=MAX_CHARGE):
        from scipy.signal import find_peaks
        data_mins =max_charge- data.min(axis=1)
        x_min = data_mins.iloc[:128]
        y_min = data_mins.iloc[128:]   

        xp,xh = find_peaks(x_min,height=0.075)
        yp,yh = find_peaks(y_min,height=0.075)

        fig,(ax1,ax2) = plt.subplots(nrows=2, ncols=1)
        fig.suptitle(f'{os.path.basename(self.data_files_list[self.current_file])}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('charge')
        ax1.grid(True)
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax1.tick_params(axis='x', labelsize=5)    

        ax1.plot(xp,xh['peak_heights'],linestyle = '',marker='.')
        ax2.plot(yp,yh['peak_heights'],linestyle = '',marker='.')
        ax2.set_xlabel('y')
        ax2.set_ylabel('charge')
        ax2.grid(True)
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax2.tick_params(axis='x', labelsize=5)
        
        fig.set_tight_layout(True)
        self.figure = fig
        self.setplot(self.figure)

    
    def detectpeaks(self):
        self.getfiles()
        data= unzipper(self.data_files_list[self.current_file])
        self.makeplot(data)

    def next_graph(self):
        if self.current_file<self.files_list_len:
            self.current_file = self.current_file+1
            data= unzipper(self.data_files_list[self.current_file])
            self.makeplot(data)

    def prev_graph(self):
        if self.current_file>0:
            self.current_file = self.current_file-1
            data= unzipper(self.data_files_list[self.current_file])
            self.makeplot(data)

    def save(self,max_charge=MAX_CHARGE,file_to_save=None):
        if file_to_save is None:
            file_to_save=self.data_files_list[self.current_file]
        
        from scipy.signal import find_peaks
        data = unzipper(file_to_save)
        data_mins =max_charge- data.min(axis=1)
        x_min = data_mins.iloc[:128]
        y_min = data_mins.iloc[128:]
        xp,xh = find_peaks(x_min,height=0.075)
        yp,yh = find_peaks(y_min,height=0.075)

        # df = pd.DataFrame.from_dict({
        #     'x_peaks':xp,
        #     'x_heights':xh['peak_heights'],
        #     'y_peaks':yp,
        #     'y_heights':yh['peak_heights']
        # },orient='index')

        df = PeakView.make_pairs(xp,xh['peak_heights'],yp,yh['peak_heights'])

        while True:
            try:
                df.to_csv(f'{os.path.basename(self.data_files_list[self.current_file]).split('.')[0]}.csv')
                break
            except PermissionError:
                retry_or = messagebox.askretrycancel("Permission error", "close open file and Try again")
            if retry_or is False:
                break




    def saveall(self,as_single_file=False):
        
        for file in self.data_files_list.values():
            self.save(file_to_save=file)
        pass

    def updategraph(self):

        width_thresh = self.width_thresh.get()
        height_thresh = self.height_thresh.get()
        max_charge = self.max_charge.get()
        print(int(width_thresh))
        print(f'{width_thresh}, {height_thresh}, {max_charge}')



class ImageReconstructionFrame(CTkFrame):
    def __init__(self, master,  **kwargs):
        super().__init__(master, **kwargs)
        self.batchsize = None
        self.processes ={'default':4,'user_set':None}
        self.hmap = None
        self.files_list = None
        self.directory = None
        self.hits_data = None
        self.width_thresh = None
        self.height_thresh = None
        self.max_charge = None

        self.plotframe = CTkFrame(master=self,width=2*SIZE[0]/3, height=SIZE[1])
        self.plotframe.grid(row=0,column = 0,columnspan =3,sticky='nsew')

        self.buttonframe = CTkFrame(master=self,width  = SIZE[0]/3, height = SIZE[1])
        self.buttonframe.grid(row=0,column = 3,sticky='nsew')

        self.button_select = ctk.CTkButton(master=self.buttonframe, text='open folder',command=self.getfiles)
        self.button_select.grid(row= 0,column= 0, padx=10,pady= 20,sticky='ew',columnspan=2)#pack(expand=False,fill='both',padx=(100,100),pady=10)

        self.button_save = ctk.CTkButton(master=self.buttonframe, text='savefig',command=self.savefig)
        self.button_save.grid(row=2,column=0, padx=10,pady= 20)#pack(expand=False,fill='both',padx=(100,100),pady=50)


        self.max_charge_label = ctk.CTkLabel(master=self.buttonframe,text= "Max charge")
        self.max_charge_label.grid(row=3,column=0, padx=1,pady= 2)

        self.max_charge = ctk.CTkEntry(master=self.buttonframe, placeholder_text= 0.125)
        self.max_charge.grid(row=3,column=1,columnspan=2, padx=1,pady= 2)

        self.height_thresh_label = ctk.CTkLabel(master=self.buttonframe,text= "Height threshold: ")
        self.height_thresh_label.grid(row=4,column=0, padx=1,pady= 2)

        self.height_thresh = ctk.CTkEntry(master=self.buttonframe, placeholder_text= 0.075)
        self.height_thresh.grid(row=4,column=1,columnspan=2, padx=1,pady= 2)

        self.width_thresh_label = ctk.CTkLabel(master=self.buttonframe,text= "Peak Width threshold(max) :")
        self.width_thresh_label.grid(row=5,column=0, padx=1,pady= 2)

        self.width_thresh = ctk.CTkEntry(master=self.buttonframe, placeholder_text= 8)
        self.width_thresh.grid(row=5,column=1,columnspan=2, padx=1,pady= 2)

        self.button_update = ctk.CTkButton(master=self.buttonframe, text='update',command=self.updategraph)
        self.button_update.grid(row=6,column=0, padx=10,pady= 20,columnspan=2)#pack(expand=False,fill='both',padx=(100,100),pady=50)

        print(self.master.master.tab('File Inspect').winfo_children())

        self.load_defaults()

    def getfiles(self):
        self.directory = filedialog.askdirectory()
        if len(self.directory) == 0:
            return
        self.data_files_list= {i:k for i,k in enumerate(glob.glob(os.path.join(self.directory,'*.zip')))}
        self.analyse()

    def plot_hitmap(self,data):
        cind = np.arange(0,128)
        dataf = pd.DataFrame(np.round((data.T)/10), columns=cind ,index=cind)
        
        self.hmap = sns.heatmap(dataf).get_figure()

        density_data = data.copy()
        # plt.contourf(np.log(data))
        # data.to_csv('img_data.csv')        
        self.setplot(self.hmap)


    def setplot(self,fig):
        plt.close()
        for wids in self.plotframe.winfo_children():
                wids.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.plotframe)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=0,column=0)    	
        toolbar = NavigationToolbar2Tk(canvas, self.plotframe, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row = 1,column= 0,sticky='ew')
    
    def analyse(self):
        if self.processes['user_set'] is None:
            n_cores = 4
        else:
            n_cores = self.processes['user_set']
        self.hits_data = analysis6mzip(core_analyser, files=list(self.data_files_list.values()),n_cores = n_cores)
        self.plot_hitmap(self.hits_data)
        
    def savefig(self):
        loca = filedialog.asksaveasfilename(filetypes=[('JPEG image', '.jpg'), ('PNG image', '.png')],                                                       
                                            defaultextension='.png',
                                            initialfile='recon_image.png'
                                            )
        print(loca)
        if len(loca) == 0:
            return
        
        self.hmap.savefig(os.path.join(loca,'recon_image.jpg'),dpi=900)
    
    def updategraph(self):

        self.width_thresh = self.width_thresh.get()
        self.height_thresh = self.height_thresh.get()
        self.max_charge = self.max_charge.get()
    

    def load_defaults(self):
        self.width_thresh = 8
        self.height_thresh = .075
        self.max_charge = .125
        self.processes = 4
        
    

class TabWrapper(ctk.CTkTabview):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.add('File Inspect')
        self.tab1 = Tab1Frame(self.tab('File Inspect'))
        self.tab1.pack(expand=True, fill = 'both')

        self.add('Peak Inspect/detect')
        self.peakanalysis = PeakView(self.tab('Peak Inspect/detect'))
        self.peakanalysis.pack(expand=True,fill='both')

        self.add('Imaging')
        self.imagerecons = ImageReconstructionFrame(self.tab('Imaging'))
        self.imagerecons.pack(expand=True,fill='both')
        self.add('settings')
        self.settings = SettingsFrame(self.tab('settings'))
        self.settings.pack(expand=True,fill='both')



class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("GEM Imaging")
        self.geometry(f'{SIZE[0]}x{SIZE[1]}')
        self.iconbitmap('favicon.ico')
        self.view = TabWrapper(self)
        self.view.pack(expand=True, fill = 'both')

        self.view.set('Peak Inspect/detect')
        #self.resizable(width=False, height=False)

if __name__ == '__main__':
    app = App()
    app.mainloop()