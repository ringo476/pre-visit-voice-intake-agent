"""track when a question was actually spoken

Revision ID: c3d1f5a92b10
Revises: 7a747524d68e
Create Date: 2026-10-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c3d1f5a92b10'
down_revision: Union[str, Sequence[str], None] = '7a747524d68e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('question_events', sa.Column('asked_in_turn', sa.Integer(), nullable=True))
    op.add_column('question_events', sa.Column('spoken_text', sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('question_events', 'spoken_text')
    op.drop_column('question_events', 'asked_in_turn')
