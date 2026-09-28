# utils/async_helpers.py
"""
Helper per eseguire coroutine async all'interno di Streamlit.

Streamlit gira il codice in un thread che ha già un event loop attivo.
Chiamare `asyncio.run()` direttamente da lì solleva RuntimeError.
Soluzione: eseguiamo la coroutine in un thread separato con il suo loop.
"""
import asyncio
import threading


def esegui_coroutine(coro):
    """
    Esegue una coroutine da un contesto sincrono (es. Streamlit).

    Se non c'è un loop attivo nel thread corrente → usa asyncio.run().
    Se c'è già un loop attivo → esegue in un thread separato.

    Args:
        coro: la coroutine da eseguire

    Returns:
        Il valore restituito dalla coroutine
    """
    # Prova a capire se c'è un event loop in esecuzione
    try:
        loop = asyncio.get_running_loop()
        loop_attivo = loop.is_running()
    except RuntimeError:
        loop_attivo = False

    if not loop_attivo:
        # Nessun loop attivo: possiamo usare asyncio.run direttamente
        return asyncio.run(coro)

    # C'è un loop attivo (Streamlit): eseguiamo in un thread dedicato
    result_container = {"value": None, "error": None}

    def _run_in_thread():
        try:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result_container["value"] = new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        except Exception as e:
            result_container["error"] = e

    thread = threading.Thread(target=_run_in_thread)
    thread.start()
    thread.join()

    if result_container["error"] is not None:
        raise result_container["error"]

    return result_container["value"]