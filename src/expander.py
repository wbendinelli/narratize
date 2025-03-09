from openai import OpenAI

client = OpenAI()
import logging

class TextExpander:
    """
    Expande resumos estruturados em capítulos completos usando a OpenAI.
    """

    def __init__(self, model="gpt-4-turbo", temperature=0.7, max_tokens=1500, verbose=True):
        """
        Inicializa o TextExpander.

        Args:
            model (str): Modelo da OpenAI a ser usado ("gpt-4-turbo").
            temperature (float): Criatividade do modelo.
            max_tokens (int): Número máximo de tokens na saída.
            verbose (bool): Se True, ativa logs detalhados.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose

        # Configuração de logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - [%(levelname)s] %(message)s"))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

    def expand_text(self, summary_text):
        """
        Expande um resumo estruturado em um texto mais longo.

        Args:
            summary_text (str): Resumo estruturado para expansão.

        Returns:
            str: Capítulo expandido.
        """
        prompt = f"""
        Expanda este resumo estruturado em um capítulo detalhado, bem escrito e fluido:

        {summary_text}

        A saída deve conter:
        - Introdução envolvente
        - Desenvolvimento claro e aprofundado de cada ponto
        - Exemplos e histórias reais
        - Conclusão inspiradora e motivacional
        """

        response = client.chat.completions.create(model=self.model,
        messages=[{"role": "system", "content": "Você é um escritor profissional especializado em transformar resumos em textos detalhados."},
                  {"role": "user", "content": prompt}],
        temperature=self.temperature,
        max_tokens=self.max_tokens)

        return response.choices[0].message.content