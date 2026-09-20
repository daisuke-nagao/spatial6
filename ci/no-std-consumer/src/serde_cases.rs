use core::fmt;

use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::ser::{
    self, SerializeMap, SerializeSeq, SerializeStruct, SerializeTuple, SerializeTupleStruct,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use core::convert::TryFrom;

use spatial6::{
    ArticulatedBodyInertia, ForceVector, MotionSubspace, MotionVector, RigidBodyInertia,
    SpatialRepresentation, SpatialScalar, SpatialTransform,
};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Error {
    Capacity,
    EndOfInput,
    InvalidToken,
    Unsupported,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Capacity => "fixed token buffer is full",
            Self::EndOfInput => "unexpected end of fixed token buffer",
            Self::InvalidToken => "invalid fixed token",
            Self::Unsupported => "unsupported fixed token operation",
        })
    }
}

impl ser::Error for Error {
    fn custom<T: fmt::Display>(_message: T) -> Self {
        Self::Unsupported
    }
}

impl de::Error for Error {
    fn custom<T: fmt::Display>(_message: T) -> Self {
        Self::InvalidToken
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Token {
    BeginSequence(usize),
    EndSequence,
    BeginMap(usize),
    EndMap,
    Number(f64),
    Bool(bool),
    Unit,
    Text { bytes: [u8; 32], length: u8 },
}

#[derive(Clone, Copy)]
struct TokenBuffer<const CAPACITY: usize> {
    tokens: [Token; CAPACITY],
    length: usize,
    human_readable: bool,
}

impl<const CAPACITY: usize> TokenBuffer<CAPACITY> {
    fn new(human_readable: bool) -> Self {
        Self {
            tokens: [Token::Unit; CAPACITY],
            length: 0,
            human_readable,
        }
    }

    fn push(&mut self, token: Token) -> Result<(), Error> {
        if self.length == CAPACITY {
            return Err(Error::Capacity);
        }
        self.tokens[self.length] = token;
        self.length += 1;
        Ok(())
    }

    fn as_slice(&self) -> &[Token] {
        &self.tokens[..self.length]
    }

    fn insert(&mut self, index: usize, token: Token) -> Result<(), Error> {
        if index > self.length || self.length == CAPACITY {
            return Err(Error::Capacity);
        }
        for position in (index..self.length).rev() {
            self.tokens[position + 1] = self.tokens[position];
        }
        self.tokens[index] = token;
        self.length += 1;
        Ok(())
    }

    fn remove(&mut self, index: usize) -> Result<Token, Error> {
        if index >= self.length {
            return Err(Error::EndOfInput);
        }
        let removed = self.tokens[index];
        for position in index..(self.length - 1) {
            self.tokens[position] = self.tokens[position + 1];
        }
        self.length -= 1;
        Ok(removed)
    }
}

struct Sequence<'a, const CAPACITY: usize> {
    buffer: &'a mut TokenBuffer<CAPACITY>,
}

impl<const CAPACITY: usize> SerializeSeq for Sequence<'_, CAPACITY> {
    type Ok = ();
    type Error = Error;

    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Error> {
        value.serialize(&mut *self.buffer)
    }

    fn end(self) -> Result<(), Error> {
        self.buffer.push(Token::EndSequence)
    }
}

impl<const CAPACITY: usize> SerializeTuple for Sequence<'_, CAPACITY> {
    type Ok = ();
    type Error = Error;

    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Error> {
        value.serialize(&mut *self.buffer)
    }

    fn end(self) -> Result<(), Error> {
        self.buffer.push(Token::EndSequence)
    }
}

impl<const CAPACITY: usize> SerializeTupleStruct for Sequence<'_, CAPACITY> {
    type Ok = ();
    type Error = Error;

    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Error> {
        value.serialize(&mut *self.buffer)
    }

    fn end(self) -> Result<(), Error> {
        self.buffer.push(Token::EndSequence)
    }
}

struct Map<'a, const CAPACITY: usize> {
    buffer: &'a mut TokenBuffer<CAPACITY>,
}

impl<const CAPACITY: usize> SerializeMap for Map<'_, CAPACITY> {
    type Ok = ();
    type Error = Error;

    fn serialize_key<T: ?Sized + Serialize>(&mut self, key: &T) -> Result<(), Error> {
        key.serialize(&mut *self.buffer)
    }

    fn serialize_value<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Error> {
        value.serialize(&mut *self.buffer)
    }

    fn end(self) -> Result<(), Error> {
        self.buffer.push(Token::EndMap)
    }
}

impl<const CAPACITY: usize> SerializeStruct for Map<'_, CAPACITY> {
    type Ok = ();
    type Error = Error;

    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Error> {
        key.serialize(&mut *self.buffer)?;
        value.serialize(&mut *self.buffer)
    }

    fn end(self) -> Result<(), Error> {
        self.buffer.push(Token::EndMap)
    }
}

impl<'a, const CAPACITY: usize> Serializer for &'a mut TokenBuffer<CAPACITY> {
    type Ok = ();
    type Error = Error;
    type SerializeSeq = Sequence<'a, CAPACITY>;
    type SerializeTuple = Sequence<'a, CAPACITY>;
    type SerializeTupleStruct = Sequence<'a, CAPACITY>;
    type SerializeTupleVariant = ser::Impossible<(), Error>;
    type SerializeMap = Map<'a, CAPACITY>;
    type SerializeStruct = Map<'a, CAPACITY>;
    type SerializeStructVariant = ser::Impossible<(), Error>;

    fn serialize_bool(self, value: bool) -> Result<(), Error> {
        self.push(Token::Bool(value))
    }

    fn serialize_i8(self, value: i8) -> Result<(), Error> {
        self.serialize_i64(value as i64)
    }

    fn serialize_i16(self, value: i16) -> Result<(), Error> {
        self.serialize_i64(value as i64)
    }

    fn serialize_i32(self, value: i32) -> Result<(), Error> {
        self.serialize_i64(value as i64)
    }

    fn serialize_i64(self, value: i64) -> Result<(), Error> {
        self.push(Token::Number(value as f64))
    }

    fn serialize_i128(self, value: i128) -> Result<(), Error> {
        self.serialize_f64(value as f64)
    }

    fn serialize_u8(self, value: u8) -> Result<(), Error> {
        self.serialize_u64(value as u64)
    }

    fn serialize_u16(self, value: u16) -> Result<(), Error> {
        self.serialize_u64(value as u64)
    }

    fn serialize_u32(self, value: u32) -> Result<(), Error> {
        self.serialize_u64(value as u64)
    }

    fn serialize_u64(self, value: u64) -> Result<(), Error> {
        self.push(Token::Number(value as f64))
    }

    fn serialize_u128(self, value: u128) -> Result<(), Error> {
        self.serialize_f64(value as f64)
    }

    fn serialize_f32(self, value: f32) -> Result<(), Error> {
        self.serialize_f64(value as f64)
    }

    fn serialize_f64(self, value: f64) -> Result<(), Error> {
        self.push(Token::Number(value))
    }

    fn serialize_char(self, value: char) -> Result<(), Error> {
        self.serialize_str(value.encode_utf8(&mut [0; 4]))
    }

    fn serialize_str(self, value: &str) -> Result<(), Error> {
        let bytes = value.as_bytes();
        if bytes.len() > 32 {
            return Err(Error::Capacity);
        }
        let mut stored = [0; 32];
        stored[..bytes.len()].copy_from_slice(bytes);
        self.push(Token::Text {
            bytes: stored,
            length: bytes.len() as u8,
        })
    }

    fn serialize_bytes(self, value: &[u8]) -> Result<(), Error> {
        let mut sequence = self.serialize_seq(Some(value.len()))?;
        for &byte in value {
            SerializeSeq::serialize_element(&mut sequence, &byte)?;
        }
        SerializeSeq::end(sequence)
    }

    fn serialize_none(self) -> Result<(), Error> {
        self.serialize_unit()
    }

    fn serialize_some<T: ?Sized + Serialize>(self, value: &T) -> Result<(), Error> {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<(), Error> {
        self.push(Token::Unit)
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<(), Error> {
        self.serialize_unit()
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
    ) -> Result<(), Error> {
        self.serialize_str(variant)
    }

    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        value: &T,
    ) -> Result<(), Error> {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _value: &T,
    ) -> Result<(), Error> {
        Err(Error::Unsupported)
    }

    fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq, Error> {
        let length = len.ok_or(Error::Unsupported)?;
        self.push(Token::BeginSequence(length))?;
        Ok(Sequence { buffer: self })
    }

    fn serialize_tuple(self, len: usize) -> Result<Self::SerializeTuple, Error> {
        self.push(Token::BeginSequence(len))?;
        Ok(Sequence { buffer: self })
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleStruct, Error> {
        self.serialize_tuple(len)
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant, Error> {
        Err(Error::Unsupported)
    }

    fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap, Error> {
        let length = len.ok_or(Error::Unsupported)?;
        self.push(Token::BeginMap(length))?;
        Ok(Map { buffer: self })
    }

    fn serialize_struct(
        self,
        _name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStruct, Error> {
        self.push(Token::BeginMap(len))?;
        Ok(Map { buffer: self })
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant, Error> {
        Err(Error::Unsupported)
    }

    fn is_human_readable(&self) -> bool {
        self.human_readable
    }

    fn collect_str<T: ?Sized + fmt::Display>(self, _value: &T) -> Result<(), Error> {
        Err(Error::Unsupported)
    }
}

struct TokenDeserializer<'de> {
    tokens: &'de [Token],
    index: usize,
    human_readable: bool,
}

impl<'de> TokenDeserializer<'de> {
    fn new(tokens: &'de [Token], human_readable: bool) -> Self {
        Self {
            tokens,
            index: 0,
            human_readable,
        }
    }

    fn peek(&self) -> Result<Token, Error> {
        self.tokens
            .get(self.index)
            .copied()
            .ok_or(Error::EndOfInput)
    }

    fn take(&mut self) -> Result<Token, Error> {
        let token = self.peek()?;
        self.index += 1;
        Ok(token)
    }

    fn begin_sequence(&mut self) -> Result<usize, Error> {
        match self.take()? {
            Token::BeginSequence(length) => Ok(length),
            _ => Err(Error::InvalidToken),
        }
    }

    fn begin_map(&mut self) -> Result<usize, Error> {
        match self.take()? {
            Token::BeginMap(length) => Ok(length),
            _ => Err(Error::InvalidToken),
        }
    }
}

struct Tokens<'a, 'de> {
    deserializer: &'a mut TokenDeserializer<'de>,
    remaining: usize,
    sequence: bool,
}

impl<'de, 'a> SeqAccess<'de> for Tokens<'a, 'de>
where
    'de: 'a,
{
    type Error = Error;

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Error>
    where
        T: DeserializeSeed<'de>,
    {
        if self.remaining == 0 {
            if matches!(self.deserializer.peek()?, Token::EndSequence) {
                self.deserializer.take()?;
                return Ok(None);
            }
            return seed.deserialize(&mut *self.deserializer).map(Some);
        }
        if matches!(self.deserializer.peek()?, Token::EndSequence) {
            self.deserializer.take()?;
            self.remaining = 0;
            return Ok(None);
        }
        self.remaining -= 1;
        seed.deserialize(&mut *self.deserializer).map(Some)
    }
}

struct KeyDeserializer(Token);

impl<'de> Deserializer<'de> for KeyDeserializer {
    type Error = Error;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_str(visitor)
    }

    fn deserialize_str<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self.0 {
            Token::Text { bytes, length } => {
                let text = core::str::from_utf8(&bytes[..length as usize])
                    .map_err(|_| Error::InvalidToken)?;
                visitor.visit_str(text)
            }
            _ => Err(Error::InvalidToken),
        }
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char string bytes byte_buf option unit
        unit_struct newtype_struct seq tuple tuple_struct map struct enum identifier ignored_any
    }
}

impl<'de, 'a> MapAccess<'de> for Tokens<'a, 'de>
where
    'de: 'a,
{
    type Error = Error;

    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>, Error>
    where
        K: DeserializeSeed<'de>,
    {
        if self.remaining == 0 {
            return match self.deserializer.take()? {
                Token::EndMap => Ok(None),
                _ => Err(Error::InvalidToken),
            };
        }
        let key = self.deserializer.take()?;
        let Token::Text { .. } = key else {
            return Err(Error::InvalidToken);
        };
        self.remaining -= 1;
        seed.deserialize(KeyDeserializer(key)).map(Some)
    }

    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value, Error>
    where
        V: DeserializeSeed<'de>,
    {
        seed.deserialize(&mut *self.deserializer)
    }
}

impl Drop for Tokens<'_, '_> {
    fn drop(&mut self) {
        if self.sequence
            && self.remaining == 0
            && matches!(
                self.deserializer.tokens.get(self.deserializer.index),
                Some(Token::EndSequence)
            )
        {
            self.deserializer.index += 1;
        }
    }
}

impl<'de, 'a> Deserializer<'de> for &'a mut TokenDeserializer<'de>
where
    'de: 'a,
{
    type Error = Error;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self.peek()? {
            Token::Number(value) => {
                self.take()?;
                visitor.visit_f64(value)
            }
            Token::Bool(value) => {
                self.take()?;
                visitor.visit_bool(value)
            }
            Token::Unit => {
                self.take()?;
                visitor.visit_unit()
            }
            Token::Text { bytes, length } => {
                self.take()?;
                let text = core::str::from_utf8(&bytes[..length as usize])
                    .map_err(|_| Error::InvalidToken)?;
                visitor.visit_str(text)
            }
            Token::BeginSequence(_) => {
                let remaining = self.begin_sequence()?;
                visitor.visit_seq(Tokens {
                    deserializer: self,
                    remaining,
                    sequence: true,
                })
            }
            Token::BeginMap(_) => {
                let remaining = self.begin_map()?;
                visitor.visit_map(Tokens {
                    deserializer: self,
                    remaining,
                    sequence: false,
                })
            }
            Token::EndSequence | Token::EndMap => Err(Error::InvalidToken),
        }
    }

    fn deserialize_bool<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self.take()? {
            Token::Bool(value) => visitor.visit_bool(value),
            _ => Err(Error::InvalidToken),
        }
    }

    fn deserialize_i8<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_i64(visitor)
    }

    fn deserialize_i16<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_i64(visitor)
    }

    fn deserialize_i32<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_i64(visitor)
    }

    fn deserialize_i64<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self.take()? {
            Token::Number(value) if value.is_finite() && value % 1.0 == 0.0 => {
                visitor.visit_i64(value as i64)
            }
            _ => Err(Error::InvalidToken),
        }
    }

    fn deserialize_i128<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self.take()? {
            Token::Number(value) if value.is_finite() && value % 1.0 == 0.0 => {
                visitor.visit_i128(value as i128)
            }
            _ => Err(Error::InvalidToken),
        }
    }

    fn deserialize_u8<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_u64(visitor)
    }

    fn deserialize_u16<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_u64(visitor)
    }

    fn deserialize_u32<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_u64(visitor)
    }

    fn deserialize_u64<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self.take()? {
            Token::Number(value) if value.is_finite() && value >= 0.0 && value % 1.0 == 0.0 => {
                visitor.visit_u64(value as u64)
            }
            _ => Err(Error::InvalidToken),
        }
    }

    fn deserialize_u128<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self.take()? {
            Token::Number(value) if value.is_finite() && value >= 0.0 && value % 1.0 == 0.0 => {
                visitor.visit_u128(value as u128)
            }
            _ => Err(Error::InvalidToken),
        }
    }

    fn deserialize_f32<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self.take()? {
            Token::Number(value) => visitor.visit_f32(value as f32),
            _ => Err(Error::InvalidToken),
        }
    }

    fn deserialize_f64<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self.take()? {
            Token::Number(value) => visitor.visit_f64(value),
            _ => Err(Error::InvalidToken),
        }
    }

    fn deserialize_str<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self.take()? {
            Token::Text { bytes, length } => {
                let text = core::str::from_utf8(&bytes[..length as usize])
                    .map_err(|_| Error::InvalidToken)?;
                visitor.visit_str(text)
            }
            _ => Err(Error::InvalidToken),
        }
    }

    fn deserialize_string<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_str(visitor)
    }

    fn deserialize_option<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        if matches!(self.peek()?, Token::Unit) {
            self.take()?;
            visitor.visit_none()
        } else {
            visitor.visit_some(self)
        }
    }

    fn deserialize_unit<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self.take()? {
            Token::Unit => visitor.visit_unit(),
            _ => Err(Error::InvalidToken),
        }
    }

    fn deserialize_seq<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        let remaining = self.begin_sequence()?;
        visitor.visit_seq(Tokens {
            deserializer: self,
            remaining,
            sequence: true,
        })
    }

    fn deserialize_tuple<V>(self, _len: usize, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_seq(visitor)
    }

    fn deserialize_tuple_struct<V>(
        self,
        _name: &'static str,
        len: usize,
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_tuple(len, visitor)
    }

    fn deserialize_map<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        let remaining = self.begin_map()?;
        visitor.visit_map(Tokens {
            deserializer: self,
            remaining,
            sequence: false,
        })
    }

    fn deserialize_struct<V>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_map(visitor)
    }

    fn deserialize_ignored_any<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_any(visitor)
    }

    fn is_human_readable(&self) -> bool {
        self.human_readable
    }

    serde::forward_to_deserialize_any! {
        char bytes byte_buf unit_struct newtype_struct enum identifier
    }
}

fn round_trip<T>(value: &T, human_readable: bool) -> bool
where
    T: Serialize + for<'de> Deserialize<'de> + PartialEq,
{
    let mut buffer = TokenBuffer::<512>::new(human_readable);
    if value.serialize(&mut buffer).is_err() {
        return false;
    }
    let mut deserializer = TokenDeserializer::new(buffer.as_slice(), human_readable);
    let Ok(decoded) = T::deserialize(&mut deserializer) else {
        return false;
    };
    decoded == *value && deserializer.index == deserializer.tokens.len()
}

fn malformed_cases<T>(value: &T) -> bool
where
    T: Serialize + for<'de> Deserialize<'de> + PartialEq,
{
    let mut original = TokenBuffer::<512>::new(true);
    if value.serialize(&mut original).is_err() || original.length < 2 {
        return false;
    }

    let mut truncated = original;
    truncated.length -= 1;
    let mut truncated_deserializer = TokenDeserializer::new(truncated.as_slice(), true);
    let truncated_rejected = T::deserialize(&mut truncated_deserializer).is_err();

    let mut excess = original;
    let excess_added = excess
        .insert(excess.length.saturating_sub(1), Token::BeginMap(0))
        .and_then(|_| excess.insert(excess.length.saturating_sub(1), Token::EndMap))
        .is_ok();
    let mut excess_deserializer = TokenDeserializer::new(excess.as_slice(), true);
    let excess_rejected = !excess_added
        || T::deserialize(&mut excess_deserializer).is_err()
        || excess_deserializer.index != excess_deserializer.tokens.len();

    let mut malformed = original;
    malformed.tokens[0] = Token::BeginMap(0);
    let mut malformed_deserializer = TokenDeserializer::new(malformed.as_slice(), true);
    let malformed_rejected = T::deserialize(&mut malformed_deserializer).is_err();

    truncated_rejected && excess_rejected && malformed_rejected
}

fn subspace_malformed<const N: usize, T, R>(value: &MotionSubspace<N, T, R>) -> bool
where
    T: SpatialScalar + Serialize + for<'de> Deserialize<'de>,
    R: SpatialRepresentation<T>,
    MotionVector<T, R>: Serialize + for<'de> Deserialize<'de>,
    MotionSubspace<N, T, R>: Serialize + for<'de> Deserialize<'de>,
{
    let mut original = TokenBuffer::<512>::new(true);
    if value.serialize(&mut original).is_err() {
        return false;
    }
    let Some(outer_end) = original
        .as_slice()
        .iter()
        .rposition(|token| matches!(token, Token::EndSequence))
    else {
        return false;
    };

    let mut truncated = original;
    let truncated_index = if N == 0 { outer_end } else { outer_end - 1 };
    let truncated_rejected = truncated.remove(truncated_index).is_ok()
        && MotionSubspace::<N, T, R>::deserialize(&mut TokenDeserializer::new(
            truncated.as_slice(),
            true,
        ))
        .is_err();

    let mut invalid = original;
    let invalid_added = if N == 0 {
        invalid.insert(1, Token::EndMap).is_ok()
    } else {
        invalid.tokens[1] = Token::EndMap;
        true
    };
    let invalid_rejected = invalid_added
        && MotionSubspace::<N, T, R>::deserialize(&mut TokenDeserializer::new(
            invalid.as_slice(),
            true,
        ))
        .is_err();

    let first_column = if N == 0 {
        MotionVector::<T, R>::from_array([T::zero(); 6])
    } else {
        value.columns()[0]
    };
    let mut encoded_column = TokenBuffer::<64>::new(true);
    if first_column.serialize(&mut encoded_column).is_err() {
        return false;
    }
    let mut excess = original;
    let excess_added = encoded_column
        .as_slice()
        .iter()
        .copied()
        .enumerate()
        .try_for_each(|(offset, token)| excess.insert(outer_end + offset, token))
        .is_ok();
    let excess_rejected = excess_added
        && MotionSubspace::<N, T, R>::deserialize(&mut TokenDeserializer::new(
            excess.as_slice(),
            true,
        ))
        .is_err();

    truncated_rejected && invalid_rejected && excess_rejected
}

fn scalar<T: SpatialScalar>(value: f64) -> T {
    T::from(value).unwrap_or_else(T::zero)
}

fn subspace<const N: usize, T: SpatialScalar, R: SpatialRepresentation<T>>(
    seed: T,
) -> MotionSubspace<N, T, R>
where
    MotionVector<T, R>: Serialize + for<'de> Deserialize<'de>,
{
    MotionSubspace::from_columns(core::array::from_fn(|column| {
        MotionVector::from_array(core::array::from_fn(|index| {
            seed + scalar::<T>((column * 7 + index) as f64 * 0.125)
        }))
    }))
}

fn exercise<T, R>(seed: T) -> bool
where
    T: SpatialScalar + Serialize + for<'de> Deserialize<'de>,
    R: SpatialRepresentation<T>,
    RigidBodyInertia<T, R>: Serialize + for<'de> Deserialize<'de>,
    ArticulatedBodyInertia<T, R>: Serialize + for<'de> Deserialize<'de>,
    ForceVector<T, R>: Serialize + for<'de> Deserialize<'de>,
    MotionVector<T, R>: Serialize + for<'de> Deserialize<'de>,
    SpatialTransform<T, R>: Serialize + for<'de> Deserialize<'de>,
{
    let s = scalar::<T>;
    let motion = MotionVector::<T, R>::from_array([seed, s(1.0), s(-2.0), s(3.0), s(-4.0), s(5.0)]);
    let force =
        ForceVector::<T, R>::from_array([s(1.0), s(-2.0), s(3.0), s(-4.0), s(5.0), s(-6.0)]);
    let transform = SpatialTransform::<T, R>::new(
        R::rotation3_identity(),
        R::vector3_from_array([s(1.0), s(2.0), s(3.0)]),
    );
    let rigid = RigidBodyInertia::<T, R>::try_new(
        s(2.0),
        R::vector3_from_array([s(0.1), s(0.2), s(0.3)]),
        R::matrix3_from_array([
            [s(1.0), s(0.0), s(0.0)],
            [s(0.0), s(1.0), s(0.0)],
            [s(0.0), s(0.0), s(1.0)],
        ]),
    );
    let Ok(rigid) = rigid else { return false };
    let articulated = ArticulatedBodyInertia::<T, R>::try_from(&rigid);
    let Ok(articulated) = articulated else {
        return false;
    };
    let values = [
        round_trip(&motion, true),
        round_trip(&motion, false),
        malformed_cases(&motion),
        round_trip(&force, true),
        round_trip(&force, false),
        malformed_cases(&force),
        round_trip(&transform, true),
        round_trip(&transform, false),
        malformed_cases(&transform),
        round_trip(&rigid, true),
        round_trip(&rigid, false),
        malformed_cases(&rigid),
        round_trip(&articulated, true),
        round_trip(&articulated, false),
        malformed_cases(&articulated),
        round_trip(&subspace::<0, T, R>(seed), true),
        round_trip(&subspace::<0, T, R>(seed), false),
        round_trip(&subspace::<1, T, R>(seed), true),
        round_trip(&subspace::<1, T, R>(seed), false),
        round_trip(&subspace::<3, T, R>(seed), true),
        round_trip(&subspace::<3, T, R>(seed), false),
        round_trip(&subspace::<6, T, R>(seed), true),
        round_trip(&subspace::<6, T, R>(seed), false),
        round_trip(&subspace::<7, T, R>(seed), true),
        round_trip(&subspace::<7, T, R>(seed), false),
    ];
    values.into_iter().all(|value| value)
        && subspace_malformed(&subspace::<0, T, R>(seed))
        && subspace_malformed(&subspace::<7, T, R>(seed))
}

pub fn run<R>(seed: f64) -> f64
where
    R: SpatialRepresentation<f64> + SpatialRepresentation<f32>,
    RigidBodyInertia<f64, R>: Serialize + for<'de> Deserialize<'de>,
    ArticulatedBodyInertia<f64, R>: Serialize + for<'de> Deserialize<'de>,
    ForceVector<f64, R>: Serialize + for<'de> Deserialize<'de>,
    MotionVector<f64, R>: Serialize + for<'de> Deserialize<'de>,
    SpatialTransform<f64, R>: Serialize + for<'de> Deserialize<'de>,
    RigidBodyInertia<f32, R>: Serialize + for<'de> Deserialize<'de>,
    ArticulatedBodyInertia<f32, R>: Serialize + for<'de> Deserialize<'de>,
    ForceVector<f32, R>: Serialize + for<'de> Deserialize<'de>,
    MotionVector<f32, R>: Serialize + for<'de> Deserialize<'de>,
    SpatialTransform<f32, R>: Serialize + for<'de> Deserialize<'de>,
{
    if exercise::<f64, R>(seed) && exercise::<f32, R>(seed as f32) {
        1.0
    } else {
        f64::NAN
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom::ArrayRepresentation;

    #[test]
    fn adapter_covers_each_supported_value() {
        let motion = MotionVector::<f64, ArrayRepresentation>::from_array([0.375; 6]);
        assert!(round_trip(&motion, true), "motion human");
        assert!(round_trip(&motion, false), "motion compact");

        let force = ForceVector::<f64, ArrayRepresentation>::from_array([0.375; 6]);
        assert!(round_trip(&force, true), "force human");
        assert!(round_trip(&force, false), "force compact");

        let transform = SpatialTransform::<f64, ArrayRepresentation>::identity();
        assert!(round_trip(&transform, true), "transform human");
        assert!(round_trip(&transform, false), "transform compact");

        let rigid = RigidBodyInertia::<f64, ArrayRepresentation>::try_new(
            2.0,
            [0.1, 0.2, 0.3],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
        .unwrap();
        assert!(round_trip(&rigid, true), "rigid human");
        assert!(round_trip(&rigid, false), "rigid compact");

        let articulated =
            ArticulatedBodyInertia::<f64, ArrayRepresentation>::try_from(&rigid).unwrap();
        assert!(round_trip(&articulated, true), "articulated human");
        assert!(round_trip(&articulated, false), "articulated compact");

        let n0 = subspace::<0, f64, ArrayRepresentation>(0.375);
        assert!(round_trip(&n0, true), "n0 human");
        assert!(round_trip(&n0, false), "n0 compact");
        assert!(subspace_malformed(&n0), "n0 malformed");
        let n1 = subspace::<1, f64, ArrayRepresentation>(0.375);
        assert!(round_trip(&n1, true), "n1 human");
        assert!(round_trip(&n1, false), "n1 compact");
        assert!(subspace_malformed(&n1), "n1 malformed");
        let n3 = subspace::<3, f64, ArrayRepresentation>(0.375);
        assert!(round_trip(&n3, true), "n3 human");
        assert!(round_trip(&n3, false), "n3 compact");
        assert!(subspace_malformed(&n3), "n3 malformed");
        let n6 = subspace::<6, f64, ArrayRepresentation>(0.375);
        assert!(round_trip(&n6, true), "n6 human");
        assert!(round_trip(&n6, false), "n6 compact");
        assert!(subspace_malformed(&n6), "n6 malformed");
        let n7 = subspace::<7, f64, ArrayRepresentation>(0.375);
        assert!(round_trip(&n7, true), "n7 human");
        assert!(round_trip(&n7, false), "n7 compact");
        assert!(subspace_malformed(&n7), "n7 malformed");
    }
}
