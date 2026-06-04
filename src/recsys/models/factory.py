"""Fábrica de modelos de recomendação baseada em registro."""

from collections.abc import Callable
from typing import Any

from recsys.models.base import RecommenderModel


class ModelFactory:
    """Cria modelos de recomendação a partir de um nome.

    O registro interno é populado por decorador. Adicionar um modelo
    novo não exige tocar nesta classe: o modelo se auto-registra.
    """

    _registry: dict[str, type[RecommenderModel]] = {}

    @classmethod
    def register(
        cls, name: str
    ) -> Callable[[type[RecommenderModel]], type[RecommenderModel]]:
        """Registra uma classe de modelo sob um nome.

        Args:
            name: Identificador usado depois em ``create``.

        Returns:
            Decorador que registra a classe e a devolve inalterada.
        """

        def decorator(
            model_cls: type[RecommenderModel],
        ) -> type[RecommenderModel]:
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> RecommenderModel:
        """Instancia o modelo registrado sob ``name``.

        Args:
            name: Identificador do modelo.
            **kwargs: Argumentos repassados ao construtor do modelo.

        Returns:
            Instância pronta do modelo.

        Raises:
            ValueError: Se ``name`` não estiver registrado.
        """
        if name not in cls._registry:
            disponiveis = ", ".join(cls.available()) or "nenhum"
            msg = f"Modelo '{name}' não registrado. Disponíveis: {disponiveis}."
            raise ValueError(msg)
        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        """Lista os modelos registrados, em ordem alfabética.

        Returns:
            Nomes registrados ordenados.
        """
        return sorted(cls._registry)
