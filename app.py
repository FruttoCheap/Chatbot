import tkinter as tk
from functools import partial

global data


def website_pressed():
    import app_urls
    data["APP"] = app_urls
    data['label'] = "Choose a Website"


def directory_pressed():
    import app_dir
    data["APP"] = app_dir
    data['label'] = "Choose a Directory"


def go_pressed(m):
    for child in m.winfo_children():
        child.destroy()

    label = tk.Label(text=data['label'])
    label.pack()
    input = tk.Entry()
    input.pack()


if __name__ == '__main__':
    data = {}
    m = tk.Tk()
    greeting = tk.Label(text="Chatbot")
    greeting.pack()
    web_button = tk.Button(
        text="Website",
        width=25,
        height=5,
        bg="blue",
        fg="black",
        command=website_pressed
    )
    web_button.pack()
    dir_button = tk.Button(
        text="Directory",
        width=25,
        height=5,
        bg="blue",
        fg="black",
        command=directory_pressed
    )
    dir_button.pack()
    go_button = tk.Button(
        text="Go",
        width=25,
        height=2,
        bg="blue",
        fg="black",
        command=partial(go_pressed, m)
    )
    go_button.pack()
    m.mainloop()
