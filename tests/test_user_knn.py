from recsys.models.factory import ModelFactory
from recsys.models.user_knn import UserKNNRecommender

# Usuários 1 e 2 compartilham os itens 10 e 20; o item 40 é exclusivo
# do usuário 2 e deve ser a primeira recomendação para o usuário 1.
_USERS = [1, 1, 1, 2, 2, 2, 3, 3, 4]
_ITEMS = [10, 20, 30, 10, 20, 40, 50, 60, 10]


def _modelo_treinado(n_neighbors: int = 2) -> UserKNNRecommender:
    model = ModelFactory.create("user_knn", n_neighbors=n_neighbors)
    model.fit(user_ids=_USERS, item_ids=_ITEMS)
    return model


def test_registrado_na_factory():
    assert "user_knn" in ModelFactory.available()
    assert isinstance(ModelFactory.create("user_knn"), UserKNNRecommender)


def test_recomenda_item_do_vizinho_mais_similar():
    model = _modelo_treinado()
    assert model.recommend(user_id=1, k=1) == [40]


def test_nao_recomenda_itens_ja_vistos():
    model = _modelo_treinado()
    recs = model.recommend(user_id=1, k=10)
    assert not {10, 20, 30} & set(recs)


def test_cold_start_recebe_itens_populares():
    model = _modelo_treinado()
    recs = model.recommend(user_id=99, k=2)
    assert recs[0] == 10  # item mais frequente do histórico
    assert len(recs) == 2


def test_completa_lista_com_populares_sem_duplicatas():
    model = _modelo_treinado()
    recs = model.recommend(user_id=1, k=3)
    assert len(recs) == 3
    assert len(set(recs)) == 3


def test_hparams_para_logging():
    model = _modelo_treinado(n_neighbors=7)
    assert model.get_hparams() == {"n_neighbors": 7, "model_type": "user_knn"}
