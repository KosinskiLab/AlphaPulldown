import numpy as np

from alphapulldown.utils import msa_encoding


def test_get_id_to_char_map_includes_gap_and_unknown():
    mapping = msa_encoding.get_id_to_char_map()

    assert mapping[21] == "-"
    assert mapping[20] == "X"


def test_get_char_to_id_map_inverts_gap_and_standard_residue():
    id_to_char = msa_encoding.get_id_to_char_map()
    char_to_id = msa_encoding.get_char_to_id_map()

    assert char_to_id["-"] == 21
    assert char_to_id[id_to_char[0]] == 0


def test_ids_to_a3m_emits_expected_headers_and_rows():
    rows = np.asarray([[0, 6, 21], [20, 5, 4]], dtype=np.int32)
    id_to_char = msa_encoding.get_id_to_char_map()

    text = msa_encoding.ids_to_a3m(rows)

    expected = (
        ">sequence_0\n"
        f"{id_to_char[0]}{id_to_char[6]}-\n"
        ">sequence_1\n"
        f"{id_to_char[20]}{id_to_char[5]}{id_to_char[4]}\n"
    )
    assert text == expected


def test_a3m_to_ids_strips_insertions_and_maps_unknowns():
    char_to_id = msa_encoding.get_char_to_id_map()
    a3m = ">sequence_0\nAcd-\n>sequence_1\nZx-\n"

    rows = msa_encoding.a3m_to_ids(a3m)

    expected = np.asarray(
        [
            [char_to_id["A"], char_to_id["-"]],
            [20, char_to_id["-"]],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(rows, expected)


def test_a3m_to_ids_returns_empty_matrix_for_empty_input():
    rows = msa_encoding.a3m_to_ids("")

    assert rows.shape == (0, 0)
    assert rows.dtype == np.int32


def test_ids_to_a3m_af3_uses_af3_alphabet_and_unknown_fallback():
    rows = np.asarray([[0, 21, 22, 29, 30, 99]], dtype=np.int32)

    text = msa_encoding.ids_to_a3m_af3(rows)

    assert text == ">sequence_0\nA-ATNX\n"
