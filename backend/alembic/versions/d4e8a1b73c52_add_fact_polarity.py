"""store the direction of each answer (present / absent / unknown) on a fact

Revision ID: d4e8a1b73c52
Revises: c3d1f5a92b10
Create Date: 2026-10-02 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4e8a1b73c52'
down_revision: Union[str, Sequence[str], None] = 'c3d1f5a92b10'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('facts', sa.Column('polarity', sa.String(), nullable=False, server_default='present'))
    # Rows saved before this existed: a denial or an "I don't know" already says its direction.
    op.execute("UPDATE facts SET polarity = 'absent' WHERE source = 'asked_and_denied'")
    op.execute("UPDATE facts SET polarity = 'unknown' WHERE source = 'uncertain'")


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('facts', 'polarity')
