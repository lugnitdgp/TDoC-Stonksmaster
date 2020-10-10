import tkinter as tk
from tkinter import messagebox as mb
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter.ttk import Combobox
from app import app

TICKER_MAPPING = {'Microsoft': 'MSFT', 'Apple': 'AAPL', 'Netflix': 'NFLX', 'Tata Motors': 'TTM', 'NVIDIA': 'NVDA', 'Intel': 'INTC'}


# handle click event on the make predictions button
def click_handler():
    # check if a stock has been selected
    combobox_val = cb.get()

    # if no, then show error popup
    if combobox_val == '' or combobox_val not in TICKER_MAPPING:
        mb.showerror('An Error Occurred!', 'Please Select a Valid Stock from the Dropdown...')
    else:
        # else plot predictions for the selected stock
        plot(TICKER_MAPPING[combobox_val])


# plot function is created for plotting the graph in tkinter window
def plot(ticker):
    # get dataframe to be plotted
    predictions_df = app(ticker)

    # check if any error occurred
    if isinstance(predictions_df, int):
        mb.showerror('Necessary Files not Found!', 'Please run trainer.py before executing the main file...')
        sys.exit(1)

    # create new window
    plot_window = tk.Tk()
    plot_window.title('Predictions')
    plot_window.geometry("1000x500+10+10")

    # create new plot
    figure = plt.Figure(figsize=(15, 7))
    ax = figure.add_subplot(111)
    line2 = FigureCanvasTkAgg(figure, plot_window)
    line2.get_tk_widget().pack()

    # customize and show the plot
    plot_title = f'Stock Price Predictions for {cb.get()}'
    xlabel = ''
    ylabel = 'Prices ($)'
    predictions_df.plot(title=plot_title, xlabel=xlabel, ylabel=ylabel, ax=ax, kind='line', legend=True)

    canvas = FigureCanvasTkAgg(figure, master=plot_window)
    canvas.draw()

    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().place()

    # placing the toolbar on the Tkinter window
    canvas.get_tk_widget().pack()


# the main Tkinter window
window = tk.Tk()

# setting the title
window.title('Stock Prediction using Supervised Machine Learning')

# dimensions of the main window
window.geometry("400x200+10+10")

# options to show within the dropdown
data = ['Apple', 'Microsoft', 'Netflix', 'Tata Motors', 'NVIDIA', 'Intel']

# create new drop down list
cb = Combobox(window, values=data)
cb.place(x=55, y=50)
cb.config(height=35, width=35)

# button that displays the plot
plot_button = tk.Button(master=window, command=click_handler, height=2, width=15, text="Make Predictions")

# place the button
plot_button.place(x=125, y=100)

# run the gui
window.mainloop()
