import math

from recsys.evaluation.metrics import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_todos_relevantes():
    assert precision_at_k({1, 2, 3}, [1, 2, 3], k=3) == 1.0


def test_precision_parcial():
    assert precision_at_k({1, 2}, [1, 9, 8, 7], k=4) == 0.25


def test_precision_k_zero_retorna_zero():
    assert precision_at_k({1}, [1], k=0) == 0.0


def test_recall_recupera_metade_dos_relevantes():
    assert recall_at_k({1, 2, 3, 4}, [1, 2, 9], k=3) == 0.5


def test_recall_sem_relevantes_retorna_zero():
    assert recall_at_k(set(), [1, 2], k=2) == 0.0


def test_map_penaliza_relevante_no_final():
    no_topo = map_at_k([{1}], [[1, 8, 9]], k=3)
    no_final = map_at_k([{1}], [[8, 9, 1]], k=3)
    assert no_topo == 1.0
    assert no_final < no_topo


def test_map_media_entre_usuarios():
    resultado = map_at_k([{1}, {2}], [[1], [9]], k=1)
    assert resultado == 0.5


def test_ndcg_ranking_perfeito_vale_um():
    assert ndcg_at_k({1, 2}, [1, 2, 9], k=3) == 1.0


def test_ndcg_penaliza_relevante_em_posicao_baixa():
    esperado = math.log2(2) / math.log2(4)  # DCG na posição 3 / DCG ideal
    assert math.isclose(ndcg_at_k({1}, [8, 9, 1], k=3), esperado)


def test_ndcg_sem_relevantes_retorna_zero():
    assert ndcg_at_k(set(), [1, 2], k=2) == 0.0
