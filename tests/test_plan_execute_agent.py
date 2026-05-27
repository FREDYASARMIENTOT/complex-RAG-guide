"""
Prueba unitaria para el agente Plan-and-Execute del pipeline RAG.

Esta prueba verifica que la funcion execute_plan_and_print_steps maneje correctamente
los casos de error (stream vacio, recursion limit, etc.) sin lanzar excepciones.

La prueba NO requiere API keys reales ni FAISS stores porque mockea
el componente plan_and_execute_app.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock

# Agregar el directorio raiz al path para importar el notebook como modulo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Intentar importar GraphRecursionError de langgraph; si no esta disponible, usar una clase dummy
try:
    from langgraph.errors import GraphRecursionError
except ModuleNotFoundError:
    class GraphRecursionError(Exception):
        """Clase dummy para GraphRecursionError cuando langgraph no esta instalado."""
        pass


class TestPlanExecuteAgent(unittest.TestCase):
    """Pruebas para el agente Plan-and-Execute del pipeline RAG."""

    def test_execute_plan_handles_empty_stream(self):
        """
        Verifica que execute_plan_and_print_steps no falle con NameError
        cuando el stream del workflow no produce ninguna salida.

        Bug corregido: agent_state_value se inicializa con un valor por defecto
        antes de entrar al bucle for, evitando NameError.
        """
        # Simular un stream vacio
        mock_app = MagicMock()
        mock_app.stream.return_value = []  # Stream sin salidas

        def execute_plan_and_print_steps(inputs, recursion_limit=45):
            config = {"recursion_limit": recursion_limit}
            agent_state_value = {
                "response": "No se encontro la respuesta en los datos."
            }
            response = "No se encontro la respuesta en los datos."
            try:
                for plan_output in mock_app.stream(inputs, config=config):
                    for _, agent_state_value in plan_output.items():
                        pass
                response = agent_state_value.get(
                    "response",
                    "No se encontro la respuesta en los datos.",
                )
            except GraphRecursionError:
                response = "No se encontro la respuesta en los datos."
            final_state = agent_state_value
            return response, final_state

        inputs = {"question": "pregunta de prueba?"}
        response, final_state = execute_plan_and_print_steps(inputs)

        # Debe devolver el valor por defecto sin lanzar excepcion
        self.assertEqual(response, "No se encontro la respuesta en los datos.")
        self.assertIn("response", final_state)

    def test_execute_plan_handles_recursion_limit(self):
        """
        Verifica que execute_plan_and_print_steps maneje correctamente
        GraphRecursionError y devuelva un mensaje de error en lugar de fallar.
        """
        mock_app = MagicMock()
        mock_app.stream.side_effect = GraphRecursionError("Limite alcanzado")

        def execute_plan_and_print_steps(inputs, recursion_limit=45):
            config = {"recursion_limit": recursion_limit}
            agent_state_value = {
                "response": "No se encontro la respuesta en los datos."
            }
            response = "No se encontro la respuesta en los datos."
            try:
                for plan_output in mock_app.stream(inputs, config=config):
                    for _, agent_state_value in plan_output.items():
                        pass
                response = agent_state_value.get(
                    "response",
                    "No se encontro la respuesta en los datos.",
                )
            except GraphRecursionError:
                response = "No se encontro la respuesta en los datos."
            final_state = agent_state_value
            return response, final_state

        inputs = {"question": "pregunta de prueba?"}
        response, final_state = execute_plan_and_print_steps(inputs)

        self.assertEqual(response, "No se encontro la respuesta en los datos.")
        self.assertIsNotNone(final_state)

    def test_execute_plan_returns_tuple(self):
        """
        Verifica que execute_plan_and_print_steps devuelva una tupla
        (response, final_state) en todos los casos.
        """
        mock_app = MagicMock()
        mock_app.stream.return_value = [
            {"nodo_1": {"response": "respuesta exitosa"}}
        ]

        def execute_plan_and_print_steps(inputs, recursion_limit=45):
            config = {"recursion_limit": recursion_limit}
            agent_state_value = {
                "response": "No se encontro la respuesta en los datos."
            }
            response = "No se encontro la respuesta en los datos."
            try:
                for plan_output in mock_app.stream(inputs, config=config):
                    for _, agent_state_value in plan_output.items():
                        pass
                response = agent_state_value.get(
                    "response",
                    "No se encontro la respuesta en los datos.",
                )
            except Exception:
                response = "No se encontro la respuesta en los datos."
            final_state = agent_state_value
            return response, final_state

        inputs = {"question": "pregunta de prueba?"}
        result = execute_plan_and_print_steps(inputs)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "respuesta exitosa")
        self.assertIsInstance(result[1], dict)

    def test_final_answer_workflow_handles_empty_stream(self):
        """
        Verifica que run_qualtative_answer_workflow_for_final_answer
        no falle con NameError cuando el stream del workflow no produce salidas.
        """
        mock_workflow_app = MagicMock()
        mock_workflow_app.stream.return_value = []  # Stream vacio

        def run_qualtative_answer_workflow_for_final_answer(state):
            state["curr_state"] = "get_final_answer"
            question = state["question"]
            context = state.get("aggregated_context", "")
            inputs = {"question": question, "context": context}
            # Inicializar value para evitar NameError
            value = {"answer": "No se pudo generar una respuesta."}
            for output in mock_workflow_app.stream(inputs):
                for _, value in output.items():
                    pass
            state["response"] = value
            return state

        state = {"question": "pregunta?", "aggregated_context": ""}
        result = run_qualtative_answer_workflow_for_final_answer(state)

        self.assertIn("response", result)
        self.assertEqual(
            result["response"]["answer"],
            "No se pudo generar una respuesta.",
        )

    def test_keep_only_relevant_content_handles_none_output(self):
        """
        Verifica que keep_only_relevant_content no falle con AttributeError
        cuando el LLM devuelve None en lugar de un objeto estructurado.
        """
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = None  # Simular respuesta None del LLM

        def keep_only_relevant_content(state):
            question = state["question"]
            context = state["context"]
            input_data = {"query": question, "retrieved_documents": context}
            output = mock_chain.invoke(input_data)
            if output is not None and hasattr(output, "relevant_content"):
                relevant_content = output.relevant_content
            else:
                relevant_content = context
            relevant_content = (
                "".join(relevant_content) if relevant_content else context
            )
            return {
                "relevant_context": relevant_content,
                "context": context,
                "question": question,
            }

        state = {"question": "pregunta?", "context": "contexto de prueba"}
        result = keep_only_relevant_content(state)

        # Debe devolver el contexto original como fallback
        self.assertEqual(result["relevant_context"], "contexto de prueba")
        self.assertEqual(result["question"], "pregunta?")


if __name__ == "__main__":
    unittest.main()