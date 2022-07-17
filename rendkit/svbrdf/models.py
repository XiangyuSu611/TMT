import pickle

from scipy.misc import imread
from sqlalchemy import (Column, Integer, String, Float, LargeBinary,
                         ForeignKey)
from sqlalchemy.orm import relationship

from gravel.database import Base


class SVBRDF(Base):
    __tablename__ = 'svbrdfs'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    substance_name = Column(String(100), nullable=True)
    path = Column(String(1024))

    def to_jsd(self):
        return {
            'type': 'svbrdf',
            'path': self.path,
            'id': self.name,
            'name': self.name,
            'substance_name': self.substance_name,
        }

    @classmethod
    def from_jsd(cls, jsd):
        new = cls()
        new.id = jsd['id']
        new.path = jsd['path']
        new.substance_name = jsd['substance_name']
        new.name = jsd['name']
        return new
