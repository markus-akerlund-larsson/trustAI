import tkinter as tk

def main():
    window = tk.Tk()
    window.title("Conversational Knowledge")

    label = tk.Label(window, text="Select an item:")
    label.pack(pady=10)

    listbox = tk.Listbox(window)
    listbox.insert(tk.END, "Category1")
    listbox.pack(padx=10, pady=10, fill="both", expand=True)

    window.mainloop()


if __name__ == "__main__":
    main()
