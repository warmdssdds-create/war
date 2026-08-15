from src.exchange.order_manager import OrderIntent, generate_order_link_id


def test_order_link_id_is_idempotent() -> None:
    intent = OrderIntent(symbol="ETCUSDT", side="Buy", qty=1.2, signal_ts=1720000)
    a = generate_order_link_id(intent)
    b = generate_order_link_id(intent)
    assert a == b
    assert len(a) == 32


def test_order_link_id_changes_with_fields() -> None:
    a = generate_order_link_id(OrderIntent(symbol="ETCUSDT", side="Buy", qty=1.0, signal_ts=1))
    b = generate_order_link_id(OrderIntent(symbol="ETCUSDT", side="Sell", qty=1.0, signal_ts=1))
    assert a != b
