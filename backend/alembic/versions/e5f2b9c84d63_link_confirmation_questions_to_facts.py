"""remember which recorded fact a confirmation question was about

Revision ID: e5f2b9c84d63
Revises: d4e8a1b73c52
Create Date: 2026-10-02 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e5f2b9c84d63'
down_revision: Union[str, Sequence[str], None] = 'd4e8a1b73c52'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('question_events', sa.Column('confirms_fact_id', sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('question_events', 'confirms_fact_id')
