"""
Programa de Métodos Numéricos
- Aproximaciones Sucesivas (Iteración de Punto Fijo)
- Condición de Lipschitz
- Método de Newton-Raphson
- Método de la Secante
"""

from typing import Callable, Tuple, Optional


def aproximaciones_sucesivas(
    g: Callable[[float], float],
    x0: float,
    max_iter: int = 100,
    tolerancia_error: Optional[float] = None
) -> Tuple[list, bool]:
    """
    Método de Aproximaciones Sucesivas (Iteración de Punto Fijo).
    Resuelve x = g(x).

    Args:
        g: Función de iteración g(x)
        x0: Valor inicial
        max_iter: Número máximo de iteraciones
        tolerancia_error: Error relativo % para criterio de parada (ej: 0.01 = 1%)

    Returns:
        (lista de aproximaciones, True si converge)
    """
    aprox = [x0]
    converge = False

    for i in range(max_iter - 1):
        xn = aprox[-1]
        x_sig = g(xn)
        aprox.append(x_sig)

        if tolerancia_error is not None:
            error_rel = abs(x_sig - xn) / abs(xn) * 100 if xn != 0 else abs(x_sig - xn) * 100
            if error_rel <= tolerancia_error:
                converge = True
                break
        else:
            if abs(x_sig - xn) < 1e-12:
                converge = True
                break

    return aprox, converge


def verificar_lipschitz(
    g: Callable[[float], float],
    intervalo: Tuple[float, float],
    num_puntos: int = 100
) -> Tuple[float, bool]:
    """
    Verifica la condición de Lipschitz: |g'(x)| < 1 en el intervalo.
    La constante de Lipschitz L = max|g'(x)| debe ser < 1 para convergencia garantizada.

    Args:
        g: Función de iteración (usa sympy para derivar)
        intervalo: (a, b) donde evaluar
        num_puntos: Puntos a muestrear

    Returns:
        (constante L, True si L < 1)
    """
    # Derivada numérica
    h = 1e-6
    L_max = 0.0
    a, b = intervalo
    paso = (b - a) / (num_puntos - 1) if num_puntos > 1 else 0

    for i in range(num_puntos):
        x = a + i * paso
        g_prime = (g(x + h) - g(x)) / h
        derivada = abs(g_prime)
        L_max = max(L_max, derivada)

    cumple_lipschitz = L_max < 1
    return L_max, cumple_lipschitz


def newton_raphson(
    f: Callable[[float], float],
    f_derivada: Callable[[float], float],
    x0: float,
    max_iter: int = 100,
    tolerancia_error: Optional[float] = None
) -> Tuple[list, bool]:
    """
    Método de Newton-Raphson para resolver f(x) = 0.

    Args:
        f: Función objetivo
        f_derivada: Derivada de f
        x0: Valor inicial
        max_iter: Número máximo de iteraciones
        tolerancia_error: Error relativo % (ej: 0.01 = 1%)

    Returns:
        (lista de aproximaciones, True si converge)
    """
    aprox = [x0]
    converge = False

    for i in range(max_iter - 1):
        xn = aprox[-1]
        f_val = f(xn)
        f_der = f_derivada(xn)

        if abs(f_der) < 1e-15:
            break  # Derivada nula

        x_sig = xn - f_val / f_der
        aprox.append(x_sig)

        if tolerancia_error is not None:
            error_rel = abs(x_sig - xn) / abs(xn) * 100 if xn != 0 else abs(x_sig - xn) * 100
            if error_rel <= tolerancia_error:
                converge = True
                break
        else:
            if abs(x_sig - xn) < 1e-12 or abs(f(x_sig)) < 1e-12:
                converge = True
                break

    return aprox, converge


def metodo_secante(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    max_iter: int = 100,
    tolerancia_error: Optional[float] = None
) -> Tuple[list, bool]:
    """
    Método de la Secante para resolver f(x) = 0.
    Usa x0 y x1 como valores iniciales.

    Args:
        f: Función objetivo
        x0: Primer valor inicial
        x1: Segundo valor inicial
        max_iter: Número máximo de iteraciones
        tolerancia_error: Error relativo % (ej: 0.01 = 1%)

    Returns:
        (lista de aproximaciones, True si converge)
    """
    aprox = [x0, x1]
    converge = False

    for i in range(max_iter - 2):
        xn = aprox[-1]
        xn_1 = aprox[-2]
        f_n = f(xn)
        f_n1 = f(xn_1)

        denom = f_n - f_n1
        if abs(denom) < 1e-15:
            break

        x_sig = xn - f_n * (xn - xn_1) / denom
        aprox.append(x_sig)

        if tolerancia_error is not None:
            error_rel = abs(x_sig - xn) / abs(xn) * 100 if xn != 0 else abs(x_sig - xn) * 100
            if error_rel <= tolerancia_error:
                converge = True
                break
        else:
            if abs(x_sig - xn) < 1e-12 or abs(f(x_sig)) < 1e-12:
                converge = True
                break

    return aprox, converge


def menu_interactivo():
    """Menú interactivo para configurar y ejecutar los métodos."""
    import math

    print("=" * 60)
    print("  MÉTODOS NUMÉRICOS - RESOLUCIÓN DE ECUACIONES")
    print("=" * 60)

    # --- Parámetros configurables ---
    try:
        max_iter = int(input("\nNúmero máximo de iteraciones [50]: ") or 50)
        error_input = input("Error relativo % para criterio de parada [0.01]: ") or "0.01"
        error_porcentaje = float(error_input)
    except ValueError:
        max_iter = 50
        error_porcentaje = 0.01
        print("Usando valores por defecto.")

    print("\n--- MÉTODOS DISPONIBLES ---")
    print("1. Aproximaciones Sucesivas (x = g(x))")
    print("2. Newton-Raphson (f(x) = 0)")
    print("3. Método de la Secante (f(x) = 0)")
    print("4. Ejecutar todos los ejemplos")
    opcion = input("\nSeleccione método [1-4]: ") or "4"

    if opcion == "1":
        x0 = float(input("Valor inicial x0 [0.5]: ") or 0.5)
        print("Ejemplo: g(x) = cos(x) para resolver x = cos(x)")
        g = lambda x: math.cos(x)
        print("\nAPROXIMACIONES SUCESIVAS (x = cos(x))")
        L, cumple = verificar_lipschitz(g, (0, 1))
        print(f"Constante de Lipschitz L ~ {L:.6f}, Converge? L<1: {'Si' if cumple else 'No'}")
        aprox, conv = aproximaciones_sucesivas(g, x0, max_iter, error_porcentaje)
        print(f"x0={x0}, Iteraciones={len(aprox)}, Raiz~{aprox[-1]:.10f}, Convergio={conv}")

    elif opcion == "2":
        x0 = float(input("Valor inicial x0 [1.5]: ") or 1.5)
        f = lambda x: x**2 - 2
        f_der = lambda x: 2 * x
        print("\nNEWTON-RAPHSON (x² - 2 = 0)")
        aprox, conv = newton_raphson(f, f_der, x0, max_iter, error_porcentaje)
        print(f"x0={x0}, Iteraciones={len(aprox)}, Raiz~{aprox[-1]:.10f}, sqrt(2)~{2**0.5:.10f}")

    elif opcion == "3":
        x0 = float(input("Primer valor inicial x0 [1.0]: ") or 1.0)
        x1 = float(input("Segundo valor inicial x1 [2.0]: ") or 2.0)
        f = lambda x: x**2 - 2
        print("\nMÉTODO DE LA SECANTE (x² - 2 = 0)")
        aprox, conv = metodo_secante(f, x0, x1, max_iter, error_porcentaje)
        print(f"x0={x0}, x1={x1}, Iteraciones={len(aprox)}, Raiz~{aprox[-1]:.10f}")

    else:
        # Ejecutar todos los ejemplos
        print("\n--- Ejecutando todos los métodos ---")
        g = lambda x: math.cos(x)
        f = lambda x: x**2 - 2
        f_der = lambda x: 2 * x

        print("\n1. APROXIMACIONES SUCESIVAS (x = cos(x))")
        L, cumple = verificar_lipschitz(g, (0, 1))
        print(f"   Lipschitz L~{L:.6f}, L<1: {'Si' if cumple else 'No'}")
        aprox, conv = aproximaciones_sucesivas(g, 0.5, max_iter, error_porcentaje)
        print(f"   x0=0.5 -> Raiz~{aprox[-1]:.10f}, iter={len(aprox)}")

        print("\n2. NEWTON-RAPHSON (x² - 2 = 0)")
        aprox, conv = newton_raphson(f, f_der, 1.5, max_iter, error_porcentaje)
        print(f"   x0=1.5 -> Raiz~{aprox[-1]:.10f}, iter={len(aprox)}")

        print("\n3. SECANTE (x² - 2 = 0)")
        aprox, conv = metodo_secante(f, 1.0, 2.0, max_iter, error_porcentaje)
        print(f"   x0=1, x1=2 -> Raiz~{aprox[-1]:.10f}, iter={len(aprox)}")

    print("\n" + "=" * 60)


def main():
    try:
        menu_interactivo()
    except KeyboardInterrupt:
        print("\n\nPrograma terminado.")


def ejecutar_ejemplo_demo():
    """Ejecuta todos los métodos con valores por defecto (sin menú)."""
    import math
    max_iter = 50
    error_porcentaje = 0.01
    g = lambda x: math.cos(x)
    f = lambda x: x**2 - 2
    f_der = lambda x: 2 * x

    print("=" * 60)
    print("  MÉTODOS NUMÉRICOS - DEMO (max_iter=50, error=0.01%)")
    print("=" * 60)

    print("\n1. APROXIMACIONES SUCESIVAS (x = cos(x))")
    L, cumple = verificar_lipschitz(g, (0, 1))
    print(f"   Lipschitz L~{L:.6f}, Converge garantizado: {'Si' if cumple else 'No'}")
    aprox, conv = aproximaciones_sucesivas(g, 0.5, max_iter, error_porcentaje)
    print(f"   x0=0.5 -> Raiz~{aprox[-1]:.10f}, Iteraciones={len(aprox)}")

    print("\n2. NEWTON-RAPHSON (x² - 2 = 0)")
    aprox, conv = newton_raphson(f, f_der, 1.5, max_iter, error_porcentaje)
    print(f"   x0=1.5 -> Raiz~{aprox[-1]:.10f}, sqrt(2)~{2**0.5:.10f}, Iteraciones={len(aprox)}")

    print("\n3. SECANTE (x² - 2 = 0)")
    aprox, conv = metodo_secante(f, 1.0, 2.0, max_iter, error_porcentaje)
    print(f"   x0=1, x1=2 -> Raiz~{aprox[-1]:.10f}, Iteraciones={len(aprox)}")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ("--gui", "-g"):
        from interfaz import AppMetodosNumericos
        app = AppMetodosNumericos()
        app.mainloop()
    elif len(sys.argv) > 1 and sys.argv[1] in ("--demo", "-d"):
        ejecutar_ejemplo_demo()
    else:
        main()
