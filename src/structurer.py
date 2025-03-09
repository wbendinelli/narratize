from openai import OpenAI

client = OpenAI()
import os
import logging
from pathlib import Path

class TextStructurer:
    """
    Processa transcrições e gera resumos estruturados utilizando a API da OpenAI.
    """

    def __init__(self, model="gpt-3.5-turbo", temperature=0.5, max_tokens=800, verbose=True):
        """
        Inicializa o TextStructurer com parâmetros ajustáveis.

        Args:
            model (str): Modelo da OpenAI a ser usado ("gpt-3.5-turbo" ou "gpt-4-turbo").
            temperature (float): Criatividade do modelo (0.0 = resposta objetiva, 1.0 = mais criativo).
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

    def _generate_prompt(self, text, style="podcast"):
        """
        Gera um prompt estruturado para a OpenAI com base no estilo escolhido.

        Args:
            text (str): Texto transcrito a ser processado.
            style (str): Estilo do resumo ("academic", "business", "podcast").

        Returns:
            str: Prompt formatado para a OpenAI.
        """
        templates = {
            "academic": f"""
            Analise o seguinte conteúdo acadêmico e crie um resumo estruturado:
            
            {text}

            Estruture a resposta da seguinte forma:

            # Introdução
            [Resumo do tema principal e contexto acadêmico]

            # Descobertas Principais
            - [Ponto 1 com evidências]
            - [Ponto 2 com evidências]
            - [Ponto 3 com evidências]

            # Aplicações Práticas
            - [Como essa pesquisa pode ser aplicada?]

            # Perguntas para Reflexão
            1. [Pergunta sobre a metodologia]
            2. [Pergunta sobre implicações futuras]
            """,

            "business": f"""
            Gere um resumo executivo deste conteúdo empresarial:

            {text}

            Estrutura do resumo:

            # Visão Geral
            [Resumo rápido do tema principal]

            # Oportunidades de Mercado
            - [Ponto chave 1]
            - [Ponto chave 2]

            # Desafios e Riscos
            - [Desafio 1]
            - [Desafio 2]

            # Métricas e Indicadores
            [Principais KPIs mencionados]

            # Conclusão e Ação Recomendável
            [Resumo final e insights principais]
            """,

            "podcast": f"""
            Estruture um resumo detalhado deste episódio de podcast:

            {text}

            Estrutura:

            # Destaques do Episódio
            - [Tópico principal abordado]
            - [Tópico secundário abordado]

            # Momentos Memoráveis
            > "Citação importante"

            # Lições e Insights
            - [Lição 1]
            - [Lição 2]

            # Recursos Citados
            - [Livros, sites ou referências mencionadas]

            # Reflexão Final
            [Resumo do episódio e impacto]
            """
        }

        return templates.get(style, templates["podcast"])

    def summarize_text(self, transcription_text, style="podcast"):
        """
        Gera um resumo estruturado usando a OpenAI.

        Args:
            transcription_text (str): Texto transcrito.
            style (str): Estilo do resumo ("academic", "business", "podcast").

        Returns:
            str: Resumo gerado.
        """
        prompt = self._generate_prompt(transcription_text, style)

        response = client.chat.completions.create(model=self.model,
        messages=[{"role": "system", "content": "Você é um especialista em análise de texto e estruturação de resumos."},
                  {"role": "user", "content": prompt}],
        temperature=self.temperature,
        max_tokens=self.max_tokens)

        return response.choices[0].message.content