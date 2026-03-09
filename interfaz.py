"""
Interfaz grafica para Metodos Numericos
- Aproximaciones Sucesivas
- Newton-Raphson
- Metodo de la Secante
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import math
from metodos_numericos import (
    aproximaciones_sucesivas,
    verificar_lipschitz,
    newton_raphson,
    metodo_secante,
)


def crear_contexto_math():
    """Contexto seguro para evaluar expresiones con x."""
    return {"math": math, "cos": math.cos, "sin": math.sin, "tan": math.tan,
            "exp": math.exp, "log": math.log, "sqrt": math.sqrt, "pi": math.pi}


def evaluar_funcion(expr: str, x: float) -> float:
    """Evalua expresion como f(x)."""
    ctx = crear_contexto_math()
    ctx["x"] = x
    return float(eval(expr, {"__builtins__": {}}, ctx))


def crear_funcion(expr: str):
    """Retorna callable f(x) a partir de expresion."""
    def f(x):
        return evaluar_funcion(expr, x)
    return f


def derivada_numerica(f, x, h=1e-6):
    """Derivada numerica de f en x."""
    return (f(x + h) - f(x - h)) / (2 * h)


class AppMetodosNumericos(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Metodos Numericos - Resolucion de Ecuaciones")
        self.geometry("700x650")
        self.configure(bg="#1e1e2e")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1e1e2e")
        style.configure("TLabel", background="#1e1e2e", foreground="#cdd6f4", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), foreground="#89b4fa")

        self.crear_widgets()

    def crear_widgets(self):
        main = ttk.Frame(self, padding=20)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="Metodos Numericos", style="Header.TLabel").pack(pady=(0, 20))

        # Metodo
        f_metodo = ttk.Frame(main)
        f_metodo.pack(fill=tk.X, pady=5)
        ttk.Label(f_metodo, text="Metodo:").pack(side=tk.LEFT, padx=(0, 10))
        self.var_metodo = tk.StringVar(value="aprox")
        self.var_metodo.trace_add("write", lambda *_: self.actualizar_campos())
        for val, txt in [("aprox", "Aproximaciones Sucesivas"), ("newton", "Newton-Raphson"), ("secante", "Secante")]:
            rb = ttk.Radiobutton(f_metodo, text=txt, variable=self.var_metodo, value=val)
            rb.pack(side=tk.LEFT, padx=5)

        # Funcion g(x) para aprox sucesivas
        self.f_g = ttk.Frame(main)
        self.f_g.pack(fill=tk.X, pady=5)
        ttk.Label(self.f_g, text="g(x) = ").pack(side=tk.LEFT, padx=(0, 5))
        self.entry_g = ttk.Entry(self.f_g, width=25)
        self.entry_g.pack(side=tk.LEFT)
        self.entry_g.insert(0, "cos(x)")
        ttk.Label(self.f_g, text="  (ej: cos(x), x**2/2 + 0.5)").pack(side=tk.LEFT, padx=5)

        # Funcion f(x) para Newton y Secante
        self.f_f = ttk.Frame(main)
        self.f_f.pack(fill=tk.X, pady=5)
        ttk.Label(self.f_f, text="f(x) = ").pack(side=tk.LEFT, padx=(0, 5))
        self.entry_f = ttk.Entry(self.f_f, width=25)
        self.entry_f.pack(side=tk.LEFT)
        self.entry_f.insert(0, "x**2 - 2")
        ttk.Label(self.f_f, text="  (ej: x**2 - 2, exp(x) - 2*x)").pack(side=tk.LEFT, padx=5)

        # f'(x) solo Newton
        self.f_fder = ttk.Frame(main)
        self.f_fder.pack(fill=tk.X, pady=5)
        ttk.Label(self.f_fder, text="f'(x) = ").pack(side=tk.LEFT, padx=(0, 5))
        self.entry_fder = ttk.Entry(self.f_fder, width=25)
        self.entry_fder.pack(side=tk.LEFT)
        self.entry_fder.insert(0, "2*x")
        ttk.Label(self.f_fder, text="  (vacío = derivada numerica)").pack(side=tk.LEFT, padx=5)

        # Valores iniciales
        self.f_val = ttk.Frame(main)
        self.f_val.pack(fill=tk.X, pady=5)
        ttk.Label(self.f_val, text="x0:").pack(side=tk.LEFT, padx=(0, 5))
        self.entry_x0 = ttk.Entry(self.f_val, width=10)
        self.entry_x0.pack(side=tk.LEFT)
        self.entry_x0.insert(0, "0.5")
        ttk.Label(self.f_val, text="  x1 (solo Secante):").pack(side=tk.LEFT, padx=(15, 5))
        self.entry_x1 = ttk.Entry(self.f_val, width=10)
        self.entry_x1.pack(side=tk.LEFT)
        self.entry_x1.insert(0, "2.0")

        # Intervalo para Lipschitz
        self.f_lip = ttk.Frame(main)
        self.f_lip.pack(fill=tk.X, pady=5)
        ttk.Label(self.f_lip, text="Intervalo Lipschitz [a,b]:").pack(side=tk.LEFT, padx=(0, 5))
        self.entry_a = ttk.Entry(self.f_lip, width=6)
        self.entry_a.pack(side=tk.LEFT)
        self.entry_a.insert(0, "0")
        ttk.Label(self.f_lip, text=" , ").pack(side=tk.LEFT)
        self.entry_b = ttk.Entry(self.f_lip, width=6)
        self.entry_b.pack(side=tk.LEFT)
        self.entry_b.insert(0, "1")

        # Parametros
        self.f_param = ttk.Frame(main)
        self.f_param.pack(fill=tk.X, pady=5)
        ttk.Label(self.f_param, text="Max iteraciones:").pack(side=tk.LEFT, padx=(0, 5))
        self.entry_iter = ttk.Entry(self.f_param, width=8)
        self.entry_iter.pack(side=tk.LEFT)
        self.entry_iter.insert(0, "50")
        ttk.Label(self.f_param, text="  Error %:").pack(side=tk.LEFT, padx=(15, 5))
        self.entry_error = ttk.Entry(self.f_param, width=8)
        self.entry_error.pack(side=tk.LEFT)
        self.entry_error.insert(0, "0.01")

        # Boton
        ttk.Button(main, text="Calcular", command=self.calcular).pack(pady=15)

        # Resultados
        ttk.Label(main, text="Resultados:").pack(anchor=tk.W)
        self.txt_result = scrolledtext.ScrolledText(main, height=18, width=80, font=("Consolas", 9),
                                                     bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4")
        self.txt_result.pack(fill=tk.BOTH, expand=True, pady=5)

        self.actualizar_campos()

    def actualizar_campos(self, event=None):
        m = self.var_metodo.get()
        # Usar before= para mantener el orden correcto al reaparecer (evita que queden debajo del area de resultados)
        self.f_g.pack_forget() if m != "aprox" else self.f_g.pack(fill=tk.X, pady=5, before=self.f_val)
        self.f_f.pack_forget() if m == "aprox" else self.f_f.pack(fill=tk.X, pady=5, before=self.f_val)
        self.f_fder.pack_forget() if m != "newton" else self.f_fder.pack(fill=tk.X, pady=5, before=self.f_val)
        self.f_lip.pack_forget() if m != "aprox" else self.f_lip.pack(fill=tk.X, pady=5, before=self.f_param)
        if m == "secante":
            self.entry_x1.config(state="normal")
        else:
            self.entry_x1.config(state="normal")

    def escribir(self, texto: str):
        self.txt_result.insert(tk.END, texto + "\n")
        self.txt_result.see(tk.END)

    def calcular(self):
        self.txt_result.delete(1.0, tk.END)
        try:
            max_iter = int(self.entry_iter.get())
            error_pct = float(self.entry_error.get())
            x0 = float(self.entry_x0.get())
        except ValueError:
            messagebox.showerror("Error", "Revisa max iteraciones, error % y x0.")
            return

        metodo = self.var_metodo.get()

        if metodo == "aprox":
            try:
                expr_g = self.entry_g.get().strip()
                g = crear_funcion(expr_g)
            except Exception as e:
                messagebox.showerror("Error", f"Expresion g(x) invalida: {e}")
                return
            try:
                a, b = float(self.entry_a.get()), float(self.entry_b.get())
            except ValueError:
                a, b = 0, 1
            L, cumple = verificar_lipschitz(g, (a, b))
            self.escribir(f"APROXIMACIONES SUCESIVAS (x = g(x))")
            self.escribir(f"g(x) = {expr_g}")
            self.escribir(f"Constante Lipschitz L ~ {L:.6f}  |  L < 1: {'Si (converge)' if cumple else 'No'}")
            aprox, conv = aproximaciones_sucesivas(g, x0, max_iter, error_pct)
            self.escribir(f"x0 = {x0}  |  Iteraciones: {len(aprox)}  |  Convergio: {conv}")
            self.escribir(f"Raiz aproximada: {aprox[-1]:.12f}")
            for i, xi in enumerate(aprox[:10]):
                self.escribir(f"  x{i} = {xi:.10f}")
            if len(aprox) > 10:
                self.escribir(f"  ... ({len(aprox)} iteraciones)")

        elif metodo == "newton":
            try:
                expr_f = self.entry_f.get().strip()
                f = crear_funcion(expr_f)
            except Exception as e:
                messagebox.showerror("Error", f"Expresion f(x) invalida: {e}")
                return
            expr_fder = self.entry_fder.get().strip()
            if expr_fder:
                try:
                    f_der = crear_funcion(expr_fder)
                except Exception as e:
                    messagebox.showerror("Error", f"Expresion f'(x) invalida: {e}")
                    return
            else:
                f_der = lambda x: derivada_numerica(f, x)
            self.escribir(f"NEWTON-RAPHSON (f(x) = 0)")
            self.escribir(f"f(x) = {expr_f}")
            aprox, conv = newton_raphson(f, f_der, x0, max_iter, error_pct)
            self.escribir(f"x0 = {x0}  |  Iteraciones: {len(aprox)}  |  Convergio: {conv}")
            self.escribir(f"Raiz aproximada: {aprox[-1]:.12f}")
            for i, xi in enumerate(aprox[:10]):
                self.escribir(f"  x{i} = {xi:.10f}")
            if len(aprox) > 10:
                self.escribir(f"  ... ({len(aprox)} iteraciones)")

        else:  # secante
            try:
                x1 = float(self.entry_x1.get())
            except ValueError:
                messagebox.showerror("Error", "x1 debe ser un numero.")
                return
            try:
                expr_f = self.entry_f.get().strip()
                f = crear_funcion(expr_f)
            except Exception as e:
                messagebox.showerror("Error", f"Expresion f(x) invalida: {e}")
                return
            self.escribir(f"METODO DE LA SECANTE (f(x) = 0)")
            self.escribir(f"f(x) = {expr_f}")
            aprox, conv = metodo_secante(f, x0, x1, max_iter, error_pct)
            self.escribir(f"x0 = {x0}, x1 = {x1}  |  Iteraciones: {len(aprox)}  |  Convergio: {conv}")
            self.escribir(f"Raiz aproximada: {aprox[-1]:.12f}")
            for i, xi in enumerate(aprox[:10]):
                self.escribir(f"  x{i} = {xi:.10f}")
            if len(aprox) > 10:
                self.escribir(f"  ... ({len(aprox)} iteraciones)")


if __name__ == "__main__":
    app = AppMetodosNumericos()
    app.mainloop()
